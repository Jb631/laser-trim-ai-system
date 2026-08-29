"""Spec 3c — FocusChart: one metric per chart, in the two views that matter.

`set_series` is the UNIT view (every measurement, Rule-1 overlays, Q7).
`set_spc_series` is the LOT view added by the 2026-08-29 FOCUS/SPC redesign:
production runs in lots, so a lot — not a unit — is what goes in or out of
control. Both live here so the Model page can toggle between them on one widget.
"""
from datetime import datetime
from typing import List, Optional

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import FRACTION_METRICS, metric_label
from laser_trim_analyzer.ml.lots import MIN_LOTS_TRAIN
from laser_trim_analyzer.ml.spc import RECENT_K, SpcSeries


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
    return {
        # Lots are POSITIONS, not dates. A real date axis squashes a week of
        # daily lots into one tick and stretches a quiet month across the page;
        # the question here is "which lot", and the date rides along as a label.
        "xs": list(range(n)),
        "values": [pt.value for pt in points],
        "ucls": [pt.ucl for pt in points],
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
        "x_dates": [pt.end.strftime("%m/%d") for pt in points],
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
        self.bind("<Destroy>", self._on_destroy)
        self._style()

    def _style(self):
        ax, t = self._ax, self.theme
        ax.set_facecolor(t.CARD)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(t.TEXT_SECONDARY)
        ax.tick_params(colors=t.TEXT_SECONDARY, labelsize=9)
        ax.title.set_color(t.TEXT_PRIMARY)

    def set_series(self, metric: str, dates: List[datetime], values: List[float],
                   baseline_mean: Optional[float] = None, baseline_std: Optional[float] = None,
                   recent_batch_start: Optional[datetime] = None) -> None:
        import numpy as np
        ax, t = self._ax, self.theme
        ax.clear()
        self._style()
        ax.set_title(metric_label(metric))
        if not dates or not values:
            ax.text(0.5, 0.5, "No measurements for this metric in the selected window.",
                    transform=ax.transAxes, ha="center", va="center", color=t.TEXT_SECONDARY)
            self.canvas.draw_idle()
            return

        arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
        finite = arr[np.isfinite(arr)]
        self._median_line = None  # set by the baseline branch, drawn post-y-window

        # SPC overlays (Rule 1 only).
        if baseline_mean is not None:
            ax.axhline(baseline_mean, color=t.TEXT_SECONDARY, ls="--", lw=1, label="Baseline mean")
            if baseline_std:
                ax.axhspan(baseline_mean - 2 * baseline_std, baseline_mean + 2 * baseline_std,
                           color=t.TIER_WARNING, alpha=0.08)
                for k in (3, -3):
                    ax.axhline(baseline_mean + k * baseline_std, color=t.TIER_OOC, ls=":", lw=1,
                               label="±3σ control limit" if k == 3 else None)
        if recent_batch_start is not None:
            ax.axvspan(recent_batch_start, dates[-1], color=t.ACCENT, alpha=0.10)

        if baseline_mean is None:
            # Trend view (e.g. Smoothness): SCATTER only. Connecting clustered, gappy
            # per-unit points with a line reads as jagged noise and implies a continuity
            # that isn't there (the "charts are all over the place" complaint).
            ax.scatter(dates, values, s=12, color=t.ACCENT, alpha=0.8, label=metric_label(metric))
        else:
            # Control chart for BATCH data (2026-07-07 redesign). file_date is
            # day-granularity: a batch lands as many units on ONE x position.
            # Drawing one line through raw units produced vertical zigzags
            # inside each batch-day and long horizontal jumps between batches —
            # "vertical lines then horizontal connected lines" (James). SPC
            # treatment for grouped data instead: individuals as dots, the
            # trend as a line through DAILY MEANS, broken across gaps >14 days
            # so idle periods don't render as fake continuity.
            ax.scatter(dates, values, s=13, color=t.ACCENT, alpha=0.5,
                       label=f"{metric_label(metric)} — units")
            by_day: dict = {}
            for d, v in zip(dates, values):
                if v is None or not np.isfinite(v):
                    continue
                by_day.setdefault(d, []).append(v)
            if by_day:
                mdates_ = sorted(by_day)
                mvals: list = []
                mx: list = []
                prev = None
                for d in mdates_:
                    if prev is not None and (d - prev).days > 14:
                        mx.append(prev)          # break the line across the gap
                        mvals.append(np.nan)
                    mx.append(d)
                    # MEDIAN, not mean: one scale-corrupt unit in a batch
                    # (e.g. 2.1e8 Ω) would drag the day's mean off-scale and
                    # draw a full-height spike. The median tracks the batch;
                    # the red Rule-1 markers still disclose the outliers.
                    mvals.append(float(np.median(by_day[d])))
                    prev = d
                self._median_line = (mx, mvals)  # drawn after the y-window is known

        # ---- Robust y-window. A control chart must keep the control band and the
        # bulk of the data legible; outliers (lone OR clustered — e.g. a 929 Ω
        # scale error among 4,700s) must NOT stretch the axis and flatten
        # everything. Both modes anchor on the central bulk (10-90 pct);
        # out-of-window points are clamped to the edge and DISCLOSED below —
        # in no-baseline mode they previously vanished silently.
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
        if limits_off_scale is not None:
            ax.text(0.01, 0.97,
                    f"±3σ control limits off-scale ({t.fmt_measure(limits_off_scale[0], 3)} … "
                    f"{t.fmt_measure(limits_off_scale[1], 3)}) — baseline spans mixed history",
                    transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                    color=t.TIER_WARNING)

        # Daily median trend line. A batch-day where MOST units are corrupt has
        # an off-scale median — drawing it (even clamped) reads as a spike. The
        # line covers only in-window days; off-window days become a GAP, and
        # their story is told by the clamped markers + disclosure below.
        med = self._median_line
        if med is not None:
            y0, y1 = ax.get_ylim()
            mx, mvals = med
            mgapped = [v if (np.isfinite(v) and y0 <= v <= y1) else np.nan
                       for v in mvals]
            ax.plot(mx, mgapped, lw=1.6, marker="o", ms=4, color=t.ACCENT_HOVER,
                    alpha=0.95, label="Daily median", zorder=4)
            self._median_line = None

        # Clamp + disclose EVERY out-of-window point (both modes). With a
        # trained baseline, out-of-limit points are Rule-1 violations (red);
        # without one they're still shown/counted, in warning amber (no SPC
        # meaning, just "far outside the bulk — check the data").
        y0, y1 = ax.get_ylim()
        has_limits = baseline_mean is not None and bool(baseline_std)
        ucl = (baseline_mean + 3 * baseline_std) if has_limits else None
        lcl = (baseline_mean - 3 * baseline_std) if has_limits else None
        ox, oy, off_vals = [], [], []
        for d, v in zip(dates, values):
            if v is None or not np.isfinite(v):
                continue
            beyond_limits = has_limits and (v > ucl or v < lcl)
            beyond_window = v > y1 or v < y0
            if beyond_limits or beyond_window:
                cy = min(max(v, y0), y1)
                if cy != v:
                    off_vals.append(v)
                ox.append(d); oy.append(cy)
        mark_color = t.TIER_OOC if has_limits else t.TIER_WARNING
        if ox:
            # Named in the legend — unexplained red dots were the most alarming
            # thing on the page (live-walk finding, 2026-07-08).
            mark_label = ("Beyond ±3σ / off-scale" if has_limits
                          else "Far outside displayed range")
            ax.scatter(ox, oy, color=mark_color, s=30, zorder=5, clip_on=False,
                       label=mark_label)
        if off_vals:
            # Name how far the worst excursion actually reaches — a clamped marker
            # alone hides magnitude, which is exactly what a QA reviewer needs.
            ext = max(off_vals, key=abs)
            ax.text(0.99, 0.97, f"▲ {len(off_vals)} off-scale (max {t.fmt_measure(ext, 3)})",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    color=mark_color)
        # ---- Explicit x-window (2026-07-08). Autoscale is LAZY and, on this
        # reused axes, held the widest range ever rendered: after viewing
        # 'All' (stretched to 2015 by one stray file), switching back to
        # 365d/90d kept the decade-wide axis and bunched all data at the
        # right. The window the user picked IS the x-range — set it from the
        # data every render instead of trusting autoscale to shrink.
        d0, d1 = min(dates), max(dates)
        span = (d1 - d0)
        from datetime import timedelta as _td
        xpad = max(span * 0.02, _td(days=1))
        ax.set_xlim(d0 - xpad, d1 + xpad)
        ax.legend(loc="best", fontsize=8, facecolor=t.CARD, edgecolor=t.BORDER, labelcolor=t.TEXT_SECONDARY)
        self._fig.tight_layout()
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
            fontsize=9, wrap=True)

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
                            fontsize=8, ha="left", va="bottom", zorder=2,
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
        ax.set_xticklabels([p["x_dates"][i] for i in ticks], fontsize=8)
        for i in ticks:
            # "open" rides on the lot's own label: a free-floating legend line
            # for the hollow marker collided with the amber note (render check),
            # and the fact belongs on the lot it describes anyway.
            ax.annotate(p["n_labels"][i] + (" · open" if i == open_idx else ""),
                        xy=(xs[i], 0),
                        xycoords=("data", "axes fraction"), xytext=(0, -18),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=7, color=t.TEXT_SECONDARY, annotation_clip=False)
        ax.set_xlim(-0.6, len(xs) - 0.4)

        y0, y1 = ax.get_ylim()
        for i, note in p["labels"].items():
            # Alternate ABOVE/BELOW by lot parity so two flagged lots side by
            # side don't print their sentences on top of each other. The side is
            # chosen by room, not parity: a flagged lot is recent by definition,
            # so it sits at the right edge, and a right-hand sentence there is
            # simply cut off (seen in the render check) — the one thing this
            # annotation exists to say.
            room_right = xs[i] < len(xs) * 0.35
            dx, ha = (12, "left") if room_right else (-12, "right")
            frac_y = (values[i] - y0) / (y1 - y0) if y1 > y0 else 0.5
            dy = 26 if i % 2 == 0 else -42
            # 26pt up / 42pt down is roughly 18% / 29% of this axes' height, so
            # a point higher than ~0.70 (or lower than ~0.32) has to take the
            # other side or the sentence lands on the title / off the bottom.
            if dy > 0 and frac_y > 0.70:
                dy = -42
            elif dy < 0 and frac_y < 0.32:
                dy = 26
            ax.annotate(note, xy=(xs[i], values[i]), xytext=(dx, dy),
                        textcoords="offset points", fontsize=8, ha=ha,
                        color=t.TIER_OOC, zorder=7, annotation_clip=False,
                        bbox=dict(facecolor=t.CARD, edgecolor="none",
                                  alpha=0.8, pad=1.5),
                        arrowprops=dict(arrowstyle="-", lw=0.8, alpha=0.6,
                                        color=t.TIER_OOC, shrinkA=0, shrinkB=5))
        if p["old_ooc_count"]:
            ax.text(0.01, 0.97,
                    f"{p['old_ooc_count']} earlier out-of-control lots in this "
                    "window (unlabeled)", transform=ax.transAxes, ha="left",
                    va="top", fontsize=7.5, color=t.TIER_WARNING)
        if not judged:
            # Silence beats an invented limit: no band, no flags, and the reason
            # said out loud instead of an empty-looking chart.
            ax.text(0.5, 0.5,
                    f"not enough lot history to judge (needs {MIN_LOTS_TRAIN} lots)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9,
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
