"""Spec 3c — FocusChart: one metric per chart, in the two views that matter.

`set_series` is the UNIT view (every measurement, Rule-1 overlays, Q7).
`set_spc_series` is the LOT view added by the 2026-08-29 FOCUS/SPC redesign:
production runs in lots, so a lot — not a unit — is what goes in or out of
control. Both live here so the Model page can toggle between them on one widget.
"""
from datetime import datetime, timedelta
from math import isfinite
from typing import List, Optional

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from laser_trim_analyzer.gui.v6.chart_redraw import debounce_resize_redraws
from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import FRACTION_METRICS, metric_label
from laser_trim_analyzer.ml.lots import MIN_LOTS_TRAIN
from laser_trim_analyzer.ml.spc import RECENT_K, SpcSeries

# Off-scale ceiling/floor markers, both chart views (2026-09-24 facelift step
# 2 Task 3b). matplotlib's scatter `s` is a marker's area in points**2 (the
# same convention as markersize**2), so a circular marker's radius is
# sqrt(s)/2 points -- ~2.7pt at s=30.
_OFFSCALE_MARKER_S = 30
# Round 3: pinned at the exact edge, a marker (radius above) was clipped in
# half by the axes boundary. Inset by its own radius plus a hair of margin
# instead -- drawn via a blended transform (axes-fraction Y, points offset;
# real-date X), so the WHOLE marker renders regardless of the y-window's data
# scale, and it can never be clipped at all.
_OFFSCALE_MARKER_INSET_PT = (_OFFSCALE_MARKER_S ** 0.5) / 2.0 + 2.0
# set_series's header band (round 2, same task): the key line (what's drawn)
# and the off-scale note share ONE row, its bottom edge this far above the
# axes -- more than double the marker radius above, so a ceiling marker
# (itself clipped to the axes) can never reach it. A second, rarer line (the
# "control limits off-scale" disclosure) stacks ABOVE that row, and the title
# above both -- see set_series's own pad arithmetic.
_HEADER_LINE_CLEARANCE_PT = 6.0
_HEADER_LINE_HEIGHT_PT = 14.0
# The unit-view rolling median (round 2): a 30-day window, widened to 90 once
# the displayed range exceeds ~18 months -- a 30-day window over years of
# history is as noisy as the old daily median was. min 5 units so a handful
# of points right after a long idle stretch don't masquerade as a median: a
# consecutive-unit gap wider than this many days is drawn as a break, never
# a straight line across the idle stretch (independent of which window size
# is in use).
_ROLLING_WINDOW_SWITCH_DAYS = 548          # ~18 months
_ROLLING_MIN_UNITS = 5
_ROLLING_GAP_BREAK_DAYS = 30


def spc_draw_params(series: SpcSeries, focus_recent: int = RECENT_K) -> dict:
    """Everything needed to DRAW an SpcSeries — pure, so every surface agrees.

    The Model page's full p-chart and the FOCUS list's 260x64 sparklines both
    render from this dict. That is deliberate: the little picture in the list
    can never flag a different lot than the big chart behind it (the "three
    surfaces, three stories" failure the redesign exists to end).

    It also owns the one judgement a chart adds on top of `ml/spc.py`: WHICH
    excursions are today's news. An ooc lot inside the last `focus_recent` lots
    is what is happening now — red, and annotated with its sentence. Older ooc
    lots are context: amber dots and ONE counted line. Re-annotating months of
    history is exactly how the old page taught people to ignore red.

    Not judged (too little lot history) means no limits exist at all: no flags,
    no labels, NaN center. The caller draws bare dots and says so.
    """
    points = series.points
    n = len(points)
    cut = n - max(int(focus_recent), 0)          # first index still "recent"
    flag_idx: List[int] = []
    old_idx: List[int] = []
    if series.judged:                            # unjudged points carry no ooc
        for i, pt in enumerate(points):
            if pt.ooc:
                (flag_idx if i >= cut else old_idx).append(i)
    fraction = series.metric in FRACTION_METRICS
    # A month/day tick reads BACKWARDS when the lots span calendar years — the
    # work data's 8887 drew "08/20" and then "07/23" for a lot ELEVEN MONTHS
    # later (render check). The year earns its space only when there is one.
    date_fmt = "%m/%d" if len({pt.end.year for pt in points}) <= 1 else "%m/%d/%y"
    # A small lot's binomial band can run past 100% (se blows up as n shrinks,
    # seen on 6607) -- drawn, that reads as "more than everyone could fail".
    # Clipped for the CHART ONLY, fraction metrics only (a continuous metric's
    # ucl is a physical unit, not a rate, and must never be clamped to 1.0);
    # the ooc verdict above already compared against the real, unclamped ucl,
    # so a small lot that failed 100% still alarms. export/evidence.py:~456
    # clips its own printed column the same way.
    ucls = ([(min(pt.ucl, 1.0) if isfinite(pt.ucl) else pt.ucl) for pt in points]
            if fraction else [pt.ucl for pt in points])
    return {
        # Lots are POSITIONS, not dates. A real date axis squashes a week of
        # daily lots into one tick and stretches a quiet month across the page;
        # the question here is "which lot", and the date rides along as a label.
        "xs": list(range(n)),
        "values": [pt.value for pt in points],
        "ucls": ucls,
        # A fail RATE below baseline is good news, never an alarm, so the shaded
        # region starts at zero. A continuous metric drifts either way, so its
        # band is the real two-sided limit.
        "band_lo": ([0.0] * n if fraction else [pt.lcl for pt in points]),
        "center": series.p_base,
        "flag_idx": flag_idx,
        "old_idx": old_idx,
        "old_ooc_count": len(old_idx),
        "open_idx": next((i for i, pt in enumerate(points) if pt.is_open), None),
        "labels": {i: points[i].note for i in flag_idx if points[i].note},
        "n_labels": [f"n={pt.n}" for pt in points],
        "x_dates": [pt.end.strftime(date_fmt) for pt in points],
        "judged": series.judged,
        "fraction": fraction,
    }


class FocusChart(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color=theme.CARD, corner_radius=theme.RADIUS_MD, **kwargs)
        self.theme = theme
        self._fig = Figure(figsize=(8, 3), dpi=96, facecolor=theme.CARD)
        self._ax = self._fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self._fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True,
                                          padx=theme.SPACE_SM, pady=theme.SPACE_SM)
        # One render at the end of a resize, not one per <Configure>.
        self._redraw = debounce_resize_redraws(self.canvas)
        self.bind("<Destroy>", self._on_destroy)
        self._style()

    def _style(self):
        import matplotlib

        ax, t = self._ax, self.theme
        ax.set_facecolor(t.CARD)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(t.TEXT_SECONDARY)
        # Numbers in Plex Mono (step 1's spec; the fonts review found no chart
        # actually asked for it — tick labels were Sans everywhere). The
        # theme's own family name, not a new literal (gui/v6/font_loader.py
        # registers it with matplotlib when the bundled files load); a family
        # LIST falls back the same way font_loader's own
        # rcParams["font.family"] does -- silently, to whatever the Sans stack
        # currently resolves to -- so a machine where the bundled font failed
        # to load draws exactly the tick labels it drew before this change.
        ax.tick_params(colors=t.TEXT_SECONDARY, labelsize=t.CHART_FONT_LARGE,
                       labelfontfamily=[t.MONO_FAMILY[0], *matplotlib.rcParams["font.family"]])
        ax.title.set_color(t.TEXT_PRIMARY)

    def set_series(self, metric: str, dates: List[datetime], values: List[float],
                   baseline_mean: Optional[float] = None, baseline_std: Optional[float] = None,
                   recent_batch_start: Optional[datetime] = None,
                   default_window_days: Optional[int] = None) -> None:
        """`default_window_days` (2026-09-24 facelift step 2 Task 3b, James:
        "that chart looks horrible" on 6607's whole-history render — 487
        off-scale dots crowned the ceiling) opens the view on the newest
        `default_window_days` of whatever `dates`/`values` it is given, the
        full range when there is less. It is OPT IN and None by default:
        SmoothnessTab's embedded FocusChart states "the chart always sees
        every record", so only a caller that wants the shorter default
        passes one. No page does today: the Model page's Units view shows
        exactly what its 30d/90d/365d/All window control loaded (facelift F4
        removed the 12-month framing it passed at the default 90-day window,
        which could never trim anything). Kept, and tested, for a caller that
        does want it.

        Round 2 (same task, controller + James, on real 6607/8340-1 renders):
        round 1's per-CALENDAR-DAY median still zigzagged hard on a sparse
        day, reading as a solid wall over 12 months, and the "control limits
        off-scale" disclosure drew INSIDE the axes on top of the ceiling
        markers. Units are now near-invisible context dots; the one strong
        line is a TIME-based rolling median (never a straight line across an
        idle stretch); there is no legend box at all — a left-aligned title,
        one small key line under it naming only what actually got drawn, the
        off-scale note sharing that same line, and every OTHER notice either
        on that line too or its own line above it, but never inside the axes.
        """
        import numpy as np
        import matplotlib.dates as mdates
        from matplotlib.transforms import blended_transform_factory, offset_copy
        ax, t = self._ax, self.theme
        ax.clear()
        self._style()
        if not dates or not values:
            ax.set_title(metric_label(metric), loc="left", pad=10)
            ax.text(0.5, 0.5, "No measurements for this metric in the selected window.",
                    transform=ax.transAxes, ha="center", va="center", color=t.TEXT_SECONDARY)
            self.canvas.draw_idle()
            return
        if default_window_days is not None:
            newest = max(dates)
            open_cutoff = newest - timedelta(days=default_window_days)
            if min(dates) < open_cutoff:
                kept = [i for i, d in enumerate(dates) if d >= open_cutoff]
                dates = [dates[i] for i in kept]
                values = [values[i] for i in kept]

        arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
        finite = arr[np.isfinite(arr)]

        # SPC overlays (Rule 1 only). Undecorated (no `label=`) -- nothing
        # here builds a legend any more; the key line below names them.
        if baseline_mean is not None:
            ax.axhline(baseline_mean, color=t.TEXT_SECONDARY, ls="--", lw=1)
            if baseline_std:
                ax.axhspan(baseline_mean - 2 * baseline_std, baseline_mean + 2 * baseline_std,
                           color=t.TIER_WARNING, alpha=0.08)
                for k in (3, -3):
                    ax.axhline(baseline_mean + k * baseline_std, color=t.TIER_OOC, ls=":", lw=1)
        if recent_batch_start is not None:
            ax.axvspan(recent_batch_start, dates[-1], color=t.ACCENT, alpha=0.10)

        # ---- Robust y-window. Computed BEFORE anything drawn below needs to
        # know it (the rolling median clips to it; the header text needs to
        # know whether the +-3sigma band ended up on-screen at all). A control
        # chart must keep the control band and the bulk of the data legible;
        # outliers (lone OR clustered — e.g. a 929 Ω scale error among
        # 4,700s) must NOT stretch the axis and flatten everything. Both
        # modes anchor on the central bulk (10-90 pct); out-of-window points
        # are clamped to the edge and DISCLOSED below.
        lo_c, hi_c = [], []
        if finite.size:
            if finite.size >= 10:
                p_lo, p_hi = np.percentile(finite, [10, 90])
            else:
                # Tiny n: window on the in-family points (within 3 sample-σ of
                # the median) so one scale error can't stretch the axis.
                med_v = float(np.median(finite))
                sd = float(np.std(finite)) or abs(med_v) * 0.05 or 1.0
                fam = [float(v) for v in finite if abs(v - med_v) <= 3 * sd]
                p_lo, p_hi = (min(fam), max(fam)) if fam else (med_v - 1.0, med_v + 1.0)
            lo_c.append(p_lo); hi_c.append(p_hi)
        limits_off_scale = None
        if baseline_mean is not None and baseline_std:
            # Include the ±3.5σ control band in the window ONLY when it is
            # commensurate with the visible data. A baseline trained across
            # mixed historical regimes can carry a σ that dwarfs the current
            # window (8340-1: σ=1.26 vs recent spread ~0.1) — always forcing
            # the band into view locked the y-axis at ±4.8 for EVERY time
            # window, so zooming 'didn't zoom'. When the band is >6x the data
            # bulk, fit to the data and annotate the off-scale limits instead.
            band_lo = baseline_mean - 3.5 * baseline_std
            band_hi = baseline_mean + 3.5 * baseline_std
            data_span = max((hi_c[0] - lo_c[0]) if lo_c else 0.0, 1e-12)
            if (band_hi - band_lo) <= 6.0 * data_span:
                lo_c.append(band_lo); hi_c.append(band_hi)
            else:
                limits_off_scale = (baseline_mean - 3 * baseline_std,
                                    baseline_mean + 3 * baseline_std)
        elif baseline_mean is not None:
            lo_c.append(baseline_mean); hi_c.append(baseline_mean)
        if lo_c and hi_c:
            lo, hi = min(lo_c), max(hi_c)
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                if finite.size:
                    lo, hi = float(finite.min()), float(finite.max())
                if hi <= lo:
                    lo, hi = lo - 1.0, hi + 1.0
            pad = (hi - lo) * 0.08 or abs(hi) * 0.1 or 1.0
            ax.set_ylim(lo - pad, hi + pad)
        y0, y1 = ax.get_ylim()

        # ---- Units + the one strong line. Both live only with a baseline to
        # grade against; the plain trend view (e.g. Smoothness) stays simple
        # scatter, unchanged -- it never had a legend-worthy overlay to begin
        # with, so it gets no key line either.
        median_drawn = False
        roll_days = None
        if baseline_mean is None:
            # Trend view: SCATTER only. Connecting clustered, gappy per-unit
            # points with a line reads as jagged noise and implies a
            # continuity that isn't there (the "charts are all over the
            # place" complaint).
            ax.scatter(dates, values, s=12, color=t.ACCENT, alpha=0.8)
        else:
            # Near-invisible dots -- context, not ink (round 2: round 1's
            # alpha=0.22 still let a sparse-day model's daily-median line read
            # as a solid teal wall over 12 months). Every unit still draws;
            # none are dropped, just faint.
            ax.scatter(dates, values, s=5, color=t.ACCENT, alpha=0.16, edgecolors="none", zorder=2)

            # TIME-based rolling median (round 2), replacing the old
            # per-CALENDAR-DAY median, which zigzagged hard whenever a day
            # carried few units — exactly what James flagged. A pandas
            # offset-string rolling window looks back a span of WALL-CLOCK
            # TIME regardless of how many (or how few) units landed on any
            # one day, which is what actually smooths a sparse model.
            # Widened to 90 days once the shown window exceeds ~18 months —
            # 30 days of noise over years of history is the same zigzag
            # again, just at a different scale. Inputs are clipped to the
            # y-window first: the median is rank-based so this rarely moves
            # it, but it guarantees the drawn line can never itself need an
            # off-scale clamp.
            import pandas as pd
            pairs = sorted((d, min(max(v, y0), y1)) for d, v in zip(dates, values)
                           if v is not None and np.isfinite(v))
            if pairs:
                window_span_days = (max(dates) - min(dates)).days
                roll_days = 90 if window_span_days > _ROLLING_WINDOW_SWITCH_DAYS else 30
                s_in = pd.Series([v for _, v in pairs],
                                 index=pd.DatetimeIndex([d for d, _ in pairs]))
                # Evaluated per UNIT (min_periods counts real units, not
                # days -- "min 5 units" per the brief) but DRAWN per DAY
                # (round 3, James: the line smeared vertically wherever a
                # day carried many units, because evaluating it at every one
                # of them nudges the window's exact row membership each
                # time, even though they share a timestamp). The window as
                # of a day's LAST unit already includes every unit from that
                # whole day, so keeping only the last per-day row is the
                # correctly-computed value with the same-day zigzag simply
                # never drawn -- not a different (and weaker) once-a-day
                # computation, the same one, sampled once.
                roll_per_unit = s_in.rolling(f"{roll_days}D",
                                             min_periods=_ROLLING_MIN_UNITS).median()
                roll = roll_per_unit.groupby(roll_per_unit.index.normalize()).last()
                mx: list = []; mvals: list = []
                prev_d = None
                for d, v in roll.items():
                    # Never drawn across a gap: a resumed model after an idle
                    # stretch gets a break, not a straight line to its first
                    # point back (2016 -> 2023 on 6607 must read empty) --
                    # independent of which rolling window size is in use above.
                    if prev_d is not None and (d - prev_d).days > _ROLLING_GAP_BREAK_DAYS:
                        mx.append(prev_d); mvals.append(float("nan"))
                    mx.append(d); mvals.append(float(v) if np.isfinite(v) else float("nan"))
                    prev_d = d
                median_drawn = any(np.isfinite(v) for v in mvals)
                if median_drawn:
                    ax.plot(mx, mvals, lw=2.2, color=t.ACCENT, zorder=4)

        # ---- In-window vs off-scale. An in-window point beyond the limits is
        # real news AT ITS OWN POSITION and is drawn there, one small dot per
        # point. A point clamped to the y-edge is off the visible scale —
        # drawing hundreds of them individually reads as a solid bar, not as
        # data (6607: 487 of them). Those are ONE marker per (calendar month,
        # top-or-bottom edge) that has any, clipped to the axes so a marker
        # can never bleed into the header row above it. Both marker kinds and
        # the note use CHECK -- "something worth a look", not a tier verdict.
        has_limits = baseline_mean is not None and bool(baseline_std)
        ucl = (baseline_mean + 3 * baseline_std) if has_limits else None
        lcl = (baseline_mean - 3 * baseline_std) if has_limits else None
        in_x, in_y = [], []
        off_vals = []
        off_by_month: dict = {}        # (year, month, "top"|"bottom") -> representative date
        for d, v in zip(dates, values):
            if v is None or not np.isfinite(v):
                continue
            beyond_limits = has_limits and (v > ucl or v < lcl)
            beyond_window = v > y1 or v < y0
            if not (beyond_limits or beyond_window):
                continue
            if beyond_window:
                off_vals.append(v)
                edge = "top" if v > y1 else "bottom"
                key = (d.year, d.month, edge)
                prev = off_by_month.get(key)
                if prev is None or d > prev:         # newest day in the month marks it
                    off_by_month[key] = d
            else:
                in_x.append(d); in_y.append(v)
        if in_x:
            ax.scatter(in_x, in_y, s=9, color=t.CHECK, alpha=0.55, edgecolors="none", zorder=5)
        if off_by_month:
            tops = [d for (_y, _m, edge), d in off_by_month.items() if edge == "top"]
            bots = [d for (_y, _m, edge), d in off_by_month.items() if edge == "bottom"]
            # Round 3: pinned exactly at y1/y0 and clip_on=True (round 2) drew
            # a marker whose center sits ON the axes edge clipped in half.
            # Inset by its own radius instead, via a BLENDED transform (real
            # dates for X, axes-fraction 1.0/0.0 offset by a fixed POINTS
            # amount for Y) -- the whole triangle renders, at any y-window
            # data scale, without ever touching the boundary it would need
            # clipping at. Still clear of the header row: that row is offset
            # the OPPOSITE way (_HEADER_LINE_CLEARANCE_PT ABOVE axes-fraction
            # 1.0), so the two can only ever move apart, never collide.
            if tops:
                top_transform = blended_transform_factory(
                    ax.transData, offset_copy(ax.transAxes, fig=self._fig,
                                              y=-_OFFSCALE_MARKER_INSET_PT, units="points"))
                ax.scatter(tops, [1.0] * len(tops), color=t.CHECK, marker="^",
                          s=_OFFSCALE_MARKER_S, zorder=5, clip_on=True,
                          transform=top_transform)
            if bots:
                bottom_transform = blended_transform_factory(
                    ax.transData, offset_copy(ax.transAxes, fig=self._fig,
                                              y=_OFFSCALE_MARKER_INSET_PT, units="points"))
                ax.scatter(bots, [0.0] * len(bots), color=t.CHECK, marker="v",
                          s=_OFFSCALE_MARKER_S, zorder=5, clip_on=True,
                          transform=bottom_transform)

        note = ""
        if off_vals:
            # Name how far the worst excursion actually reaches, and which
            # edge(s) — a clamped marker alone hides magnitude. The total is
            # the true count over every point in the current window, not just
            # the (now aggregated) markers drawing it.
            ext = max(off_vals, key=abs)
            n_top = sum(1 for v in off_vals if v > y1)
            n_bottom = len(off_vals) - n_top
            if n_bottom == 0:
                where = f"{len(off_vals)} off-scale above"
            elif n_top == 0:
                where = f"{len(off_vals)} off-scale below"
            else:
                where = f"{n_top} off-scale above, {n_bottom} below"
            note = f"▲ {where} the chart (max {t.fmt_measure(ext, 3)})"

        # ---- Header: a left-aligned title, and directly under it one small
        # key line naming ONLY what actually got drawn above, the off-scale
        # note right-aligned on that same line — no legend box (round 2: the
        # old one sat on the data as often as not, and it kept naming limit
        # lines that were not even on screen once the band went off-scale).
        # Any OTHER notice (the "control limits off-scale" disclosure) gets
        # its own line ABOVE this one; nothing here is ever drawn inside the
        # axes, which is the round-2 bug on 8340-1 (it sat on the ceiling
        # markers) this replaces.
        key_bits = []
        if baseline_mean is not None:
            if median_drawn:
                key_bits.append(f"━ {roll_days}-day median")
            key_bits.append("·  units")
            if in_x:
                # The red dots are the chart's news -- a drawn marker is always named
                # (the 2026-07-08 walk found red markers nobody had explained).
                key_bits.append("●  beyond ±3σ (red)")
            if limits_off_scale is None and baseline_std:
                key_bits.append("┄ ±3σ control limit")
            # Named only when its dashed line is actually on the chart (final review,
            # 2026-09-24): a baseline trained across mixed history can sit far outside what the
            # recent data spans (8340-1), and the key used to name a line nobody could see.
            if y0 <= baseline_mean <= y1:
                key_bits.append("╌ baseline mean")
        key_text = "    ".join(key_bits)

        show_header_row = bool(key_text) or bool(note)
        if show_header_row:
            header_transform = offset_copy(ax.transAxes, fig=self._fig,
                                           y=_HEADER_LINE_CLEARANCE_PT, units="points")
            if key_text:
                ax.text(0.0, 1.0, key_text, transform=header_transform, ha="left", va="bottom",
                        clip_on=False, fontsize=t.CHART_FONT_SMALL, color=t.TEXT_SECONDARY)
            if note:
                ax.text(1.0, 1.0, note, transform=header_transform, ha="right", va="bottom",
                        clip_on=False, fontsize=t.CHART_FONT_SMALL, color=t.CHECK)
        if limits_off_scale is not None:
            # Stacked ABOVE the header row (never inside the axes -- was
            # ax.text(0.01, 0.97, ..., va="top"), drawn on top of the ceiling
            # off-scale markers on 8340-1's real render).
            notice_transform = offset_copy(
                ax.transAxes, fig=self._fig,
                y=_HEADER_LINE_CLEARANCE_PT + _HEADER_LINE_HEIGHT_PT, units="points")
            ax.text(0.0, 1.0,
                    f"±3σ control limits off-scale ({t.fmt_measure(limits_off_scale[0], 3)} … "
                    f"{t.fmt_measure(limits_off_scale[1], 3)}) — baseline spans mixed history",
                    transform=notice_transform, ha="left", va="bottom", clip_on=False,
                    fontsize=t.CHART_FONT_SMALL, color=t.TIER_WARNING)

        n_header_lines = (1 if show_header_row else 0) + (1 if limits_off_scale is not None else 0)
        title_pad = (10 if n_header_lines == 0
                    else _HEADER_LINE_CLEARANCE_PT + _HEADER_LINE_HEIGHT_PT * n_header_lines + 6)
        ax.set_title(metric_label(metric), loc="left", pad=title_pad)

        # ---- Explicit x-window (2026-07-08). Autoscale is LAZY and, on this
        # reused axes, held the widest range ever rendered: after viewing
        # 'All' (stretched to 2015 by one stray file), switching back to
        # 365d/90d kept the decade-wide axis and bunched all data at the
        # right. The window the user picked IS the x-range — set it from the
        # data every render instead of trusting autoscale to shrink.
        d0, d1 = min(dates), max(dates)
        span = (d1 - d0)
        xpad = max(span * 0.02, timedelta(days=1))
        ax.set_xlim(d0 - xpad, d1 + xpad)
        # Round 3, James: month labels ran together ("2026-022026-03") at a
        # 12-month window -- the implicit default formatter/locator packed
        # ticks too densely for the available width. AutoDateLocator picks
        # HOW MANY ticks actually fit (not a fixed one-per-month), and
        # ConciseDateFormatter drops what is already established on the
        # axis (a bare "Feb" once a year has been shown) instead of
        # repeating the full date at every tick -- both together are what
        # keeps labels apart, at a wide window or a narrow one.
        locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        self._fig.tight_layout()
        # tight_layout already makes room for the header text above (checked
        # on 8340-1's full history, both the mixed-history notice and an
        # off-scale note present: axes top landed well under this floor) --
        # this is a backstop for the rare case it doesn't (a very short
        # title, a different figsize).
        if n_header_lines and self._ax.get_position().y1 > 0.80:
            self._fig.subplots_adjust(top=0.80)
        self.canvas.draw_idle()

    def set_spc_series(self, series: SpcSeries, *,
                       focus_recent: int = RECENT_K) -> None:
        """Draw one (model, metric) as a LOT control chart — the p-chart view.

        Kept beside `set_series` rather than replacing it: this answers "is this
        LOT out of family for this model", while `set_series` still answers "what
        did each unit measure". The Model page toggles between them.

        Everything drawn here comes from `spc_draw_params`, so the chart cannot
        show a different set of flagged lots than the FOCUS list that sent the
        user here. The title carries the reading key on purpose — a shaded band
        with no explanation is the thing supervisors said they could not read.
        """
        import numpy as np
        from matplotlib.ticker import FuncFormatter

        p = spc_draw_params(series, focus_recent=focus_recent)
        ax, t = self._ax, self.theme
        ax.clear()
        self._style()
        fraction = p["fraction"]
        # The reading key, in the title, because a shaded band nobody can read
        # is the thing supervisors said defeated the old chart. "of that size"
        # is claimed ONLY for the p-chart, where the limit really does move with
        # the lot's n; a continuous band is ±3σ of lot medians and does not.
        key = ("shaded = what this model's history says a lot of that size can "
               "do by chance" if fraction else
               "shaded = what this model's history says a lot can do by chance")
        # wrap=True is load-bearing: the key line is ~110 characters and this
        # widget resizes with the window. Unwrapped it ran off BOTH edges of the
        # figure (render check) — the reading key clipped is the same as absent.
        ax.set_title(
            f"{series.model} — {metric_label(series.metric)} by production lot\n"
            f"{key} · red = beyond it: something changed",
            fontsize=t.CHART_FONT_LARGE, wrap=True)

        xs, values = p["xs"], p["values"]
        if not xs:
            ax.set_xticks([]); ax.set_yticks([])   # a 0.0-1.0 grid means nothing
            ax.text(0.5, 0.5, "No production lots for this model yet.",
                    transform=ax.transAxes, ha="center", va="center",
                    color=t.TEXT_SECONDARY)
            self._fig.tight_layout()
            self.canvas.draw_idle()
            return

        judged, center = p["judged"], p["center"]
        if judged:
            # Step, not a smooth band: the limit is recomputed from each lot's
            # OWN size, which is the whole point — 2 fails out of 5 is noise,
            # 2 out of 200 is a signal, and one flat threshold can't say both.
            ax.fill_between(xs, p["band_lo"], p["ucls"], step="mid",
                            color=t.ELEVATED, alpha=0.9, lw=0, zorder=0)
            if np.isfinite(center):
                ax.axhline(center, color=t.TEXT_SECONDARY, ls="--", lw=1, zorder=1)
                ax.annotate(f"baseline {center * 100:.0f}%" if fraction
                            else f"baseline {center:.4g}",
                            xy=(0.0, center), xycoords=("axes fraction", "data"),
                            xytext=(3, 3), textcoords="offset points",
                            fontsize=t.CHART_FONT, ha="left", va="bottom", zorder=2,
                            color=t.TEXT_SECONDARY,
                            bbox=dict(facecolor=t.CARD, edgecolor="none",
                                      alpha=0.75, pad=1.0))

        ax.plot(xs, values, "o", ms=5, ls="none", color=t.TEXT_SECONDARY, zorder=3)
        if p["old_idx"]:
            # Amber, not red: these lots ARE out of control, but they are older
            # than the window the verdict is about. Counted below, not narrated.
            ax.plot([xs[i] for i in p["old_idx"]], [values[i] for i in p["old_idx"]],
                    "o", ms=5.5, ls="none", color=t.TIER_WARNING, zorder=4)
        if p["flag_idx"]:
            ax.plot([xs[i] for i in p["flag_idx"]], [values[i] for i in p["flag_idx"]],
                    "o", ms=7, ls="none", color=t.TIER_OOC, zorder=5)
        open_idx = p["open_idx"]
        if open_idx is not None:
            # Hollow = still receiving units: a preview, not a verdict. Filled
            # with the card colour so the solid dot underneath doesn't show.
            edge = (t.TIER_OOC if open_idx in p["flag_idx"] else
                    t.TIER_WARNING if open_idx in p["old_idx"] else t.TEXT_SECONDARY)
            ax.plot([xs[open_idx]], [values[open_idx]], "o", ms=7.5, ls="none",
                    mfc=t.CARD, mec=edge, mew=1.8, zorder=6)

        # ---- y-window: the band must stay visible even when every lot is clean
        # (an all-zero fail rate would otherwise autoscale to a meaningless
        # sliver), and a 100% lot must not be clipped off the top.
        span_vals = [v for v in values if np.isfinite(v)]
        hi_c = list(span_vals)
        lo_c = list(span_vals)
        if judged:
            hi_c += [u for u in p["ucls"] if np.isfinite(u)]
            lo_c += [b for b in p["band_lo"] if np.isfinite(b)]
            if np.isfinite(center):
                hi_c.append(center); lo_c.append(center)
        hi = max(hi_c) if hi_c else 1.0
        lo = 0.0 if fraction else (min(lo_c) if lo_c else 0.0)
        if fraction:
            hi = min(max(hi, 0.05), 1.0)
        span = (hi - lo) or (abs(hi) * 0.1) or 1.0
        # Extra headroom when a sentence is annotated above a flagged lot.
        top_pad = 0.30 if p["labels"] else 0.15
        ax.set_ylim(lo - span * 0.06, hi + span * top_pad)
        ax.yaxis.set_major_formatter(FuncFormatter(
            (lambda v, _pos: f"{v * 100:.0f}%") if fraction
            else (lambda v, _pos: t.fmt_measure(v, 4))))

        # ---- x labels: lot end date, with the lot SIZE under it. n is not
        # decoration — it is why the band above that lot is the width it is.
        # Thinned from the RIGHT so the newest lot always keeps its label.
        stride = max(1, -(-len(xs) // 12))
        ticks = list(range(len(xs) - 1, -1, -stride))[::-1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([p["x_dates"][i] for i in ticks], fontsize=t.CHART_FONT)
        for i in ticks:
            # "open" rides on the lot's own label: a free-floating legend line
            # for the hollow marker collided with the amber note (render check),
            # and the fact belongs on the lot it describes anyway.
            ax.annotate(p["n_labels"][i] + (" · open" if i == open_idx else ""),
                        xy=(xs[i], 0),
                        xycoords=("data", "axes fraction"), xytext=(0, -18),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=t.CHART_FONT_SMALL, color=t.TEXT_SECONDARY, annotation_clip=False)
        ax.set_xlim(-0.6, len(xs) - 0.4)

        y0, y1 = ax.get_ylim()
        n_above = n_below = 0
        for slot, i in enumerate(sorted(p["labels"])):
            note = p["labels"][i]
            # The horizontal side is chosen by room: a flagged lot is recent by
            # definition, so it sits at the right edge, and a right-hand
            # sentence there is simply cut off (seen in the render check) — the
            # one thing this annotation exists to say.
            room_right = xs[i] < len(xs) * 0.35
            dx, ha = (12, "left") if room_right else (-12, "right")
            frac_y = (values[i] - y0) / (y1 - y0) if y1 > y0 else 0.5
            # ABOVE/BELOW alternates in LABEL order. Lot-index parity looked
            # like it did this, but two flagged lots an EVEN number of lots
            # apart drew the same offset and printed one sentence on top of the
            # other — 6607's last two excursions, both unreadable (render
            # check). Label order can't collide however the lots fall.
            above = slot % 2 == 0
            # 26pt up / 42pt down is roughly 18% / 29% of this axes' height, so
            # a point higher than ~0.70 (or lower than ~0.32) has to take the
            # other side or the sentence lands on the title / off the bottom.
            if above and frac_y > 0.70:
                above = False
            elif not above and frac_y < 0.32:
                above = True
            # A forced flip can put two sentences back on the same side; each
            # one after the first steps further out rather than overlapping.
            if above:
                dy = 26 + 20 * n_above
                n_above += 1
            else:
                dy = -42 - 20 * n_below
                n_below += 1
            ax.annotate(note, xy=(xs[i], values[i]), xytext=(dx, dy),
                        textcoords="offset points", fontsize=t.CHART_FONT, ha=ha,
                        color=t.TIER_OOC, zorder=7, annotation_clip=False,
                        bbox=dict(facecolor=t.CARD, edgecolor="none",
                                  alpha=0.8, pad=1.5),
                        arrowprops=dict(arrowstyle="-", lw=0.8, alpha=0.6,
                                        color=t.TIER_OOC, shrinkA=0, shrinkB=5))
        if p["old_ooc_count"]:
            ax.text(0.01, 0.97,
                    f"{p['old_ooc_count']} earlier out-of-control lots in this "
                    "window (unlabeled)", transform=ax.transAxes, ha="left",
                    va="top", fontsize=t.CHART_FONT_SMALL, color=t.TIER_WARNING)
        if not judged:
            # Silence beats an invented limit: no band, no flags, and the reason
            # said out loud instead of an empty-looking chart.
            ax.text(0.5, 0.5,
                    f"not enough lot history to judge (needs {MIN_LOTS_TRAIN} lots)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=t.CHART_FONT_LARGE,
                    color=t.TEXT_SECONDARY, zorder=8,
                    bbox=dict(facecolor=t.CARD, edgecolor="none", alpha=0.85))

        self._fig.tight_layout()
        # tight_layout can't see the offset n= row; reserve the space itself.
        if self._ax.get_position().y0 < 0.24:
            self._fig.subplots_adjust(bottom=0.24)
        self.canvas.draw_idle()

    def _on_destroy(self, _evt=None):
        try:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
        except Exception:
            pass
