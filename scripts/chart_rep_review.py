"""Chart-representation review (2026-07-16): current in-app views vs
proposed representations, rendered from the REAL work DB so James can judge
"best representation of the data" with his own models, side by side.

Per model, three comparisons:
  A. focus_current  — what the Model page draws today: unit dots + DAILY
     median line + baseline band.
     focus_lots     — proposal: the DETECTOR'S observations: one marker per
     LOT (median, sized by lot n), baseline mean ± 2σ/3σ from lot history,
     out-of-band lots flagged red. Units as faint background dots.
  B. fail_current   — today's daily-average fail rate line.
     fail_lots      — proposal: per-lot fail fraction, marker sized by n,
     with n-aware binomial limits around the baseline rate (p-chart logic).
  C. dist_resistance — proposal (no current equivalent): distribution of
     untrimmed vs trimmed resistance over the window — the picture behind
     "is the as-fired resistance target too low?"

Usage: .venv/bin/python scripts/chart_rep_review.py [db] [outdir]
"""
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from laser_trim_analyzer.ml.lots import cluster_lots  # noqa: E402

DB = sys.argv[1] if len(sys.argv) > 1 else "data/7-14-2026/data/analysis.db"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/chart_rep_review_2026-07-16")
MODELS = ["8232-1", "7845", "8877-4", "6607"]
WINDOW_DAYS = 365

BG = "#232a36"
FG = "#dfe6f0"
ACCENT = "#4da3ff"
DIM = "#5a6b82"
RED = "#ff5a5a"
AMBER = "#ffb84d"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
    "ytick.color": FG, "axes.edgecolor": DIM, "font.size": 10,
})


def load(db, model, metric_sql, cutoff):
    q = f"""SELECT a.file_date, {metric_sql}
            FROM track_results t JOIN analysis_results a ON a.id=t.analysis_id
            WHERE a.model=? AND a.file_date>=? AND {metric_sql} IS NOT NULL
            ORDER BY a.file_date"""
    rows = db.execute(q, (model, cutoff)).fetchall()
    return [(datetime.fromisoformat(d), v) for d, v in rows]


def daily_median(samples):
    by = {}
    for d, v in samples:
        by.setdefault(d.date(), []).append(v)
    days = sorted(by)
    return days, [float(np.median(by[d])) for d in days]


def render_focus_pair(db, model, cutoff):
    samples = load(db, model, "t.untrimmed_resistance", cutoff)
    if len(samples) < 30:
        return
    dates = [d for d, _ in samples]
    vals = [v for _, v in samples]
    lots = [l for l in cluster_lots(samples)]
    closed = [l for l in lots if not l.is_open()]
    if len(closed) < 5:
        return
    base = closed[:-3] if len(closed) > 8 else closed
    bvals = [l.median for l in base if l.n >= 3] or [l.median for l in base]
    bmean, bstd = float(np.mean(bvals)), float(np.std(bvals, ddof=1) or 1e-9)
    bstd = max(bstd, 0.01 * abs(bmean))

    # A1 — current representation
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.scatter(dates, vals, s=8, color=ACCENT, alpha=0.45, label="Units")
    dd, dm = daily_median(samples)
    ax.plot(dd, dm, color=ACCENT, lw=1.6, label="Daily median")
    ax.axhline(bmean, color=FG, ls="--", lw=1, alpha=0.7, label="Baseline mean")
    for k in (3, -3):
        ax.axhline(bmean + k * bstd, color=RED, ls=":", lw=1, alpha=0.7)
    ax.set_title(f"{model} — untrimmed resistance — CURRENT (daily median + units)")
    ax.legend(loc="best", fontsize=8, framealpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / f"A_focus_current_{model}.png", dpi=110)
    plt.close(fig)

    # A2 — proposed lot-control chart
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.scatter(dates, vals, s=6, color=DIM, alpha=0.25, label="Units (context)")
    lx = [l.end for l in lots]
    ly = [l.median for l in lots]
    ln = [l.n for l in lots]
    sizes = [max(24, min(180, n * 4)) for n in ln]
    colors = []
    for l in lots:
        if l.is_open():
            colors.append(AMBER)
        elif abs(l.median - bmean) > 3 * bstd:
            colors.append(RED)
        else:
            colors.append(ACCENT)
    ax.scatter(lx, ly, s=sizes, c=colors, edgecolors=FG, linewidths=0.6,
               zorder=5, label="Lot median (size = units in lot)")
    ax.plot(lx, ly, color=ACCENT, lw=0.8, alpha=0.5, zorder=4)
    ax.axhline(bmean, color=FG, ls="--", lw=1, alpha=0.8, label="Baseline (lot history)")
    ax.fill_between([min(dates), max(dates)], bmean - 2 * bstd, bmean + 2 * bstd,
                    color=ACCENT, alpha=0.10, label="±2σ of lot medians")
    for k in (3, -3):
        ax.axhline(bmean + k * bstd, color=RED, ls=":", lw=1, alpha=0.8)
    ax.set_title(f"{model} — untrimmed resistance — PROPOSED lot control chart "
                 f"(red = lot beyond 3σ, amber = open lot)")
    ax.legend(loc="best", fontsize=8, framealpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / f"A_focus_lots_{model}.png", dpi=110)
    plt.close(fig)


def render_fail_pair(db, model, cutoff):
    rows = db.execute(
        """SELECT file_date, CASE overall_status WHEN 'FAIL' THEN 1.0 ELSE 0.0 END
           FROM analysis_results WHERE model=? AND file_date>=?
             AND overall_status IN ('PASS','WARNING','FAIL')
           ORDER BY file_date""", (model, cutoff)).fetchall()
    samples = [(datetime.fromisoformat(d), v) for d, v in rows]
    if len(samples) < 30:
        return
    # B1 — current: daily average
    fig, ax = plt.subplots(figsize=(9, 3.8))
    dd, dm = daily_median(samples)  # median of 0/1 is jumpy; the app uses avg
    by = {}
    for d, v in samples:
        by.setdefault(d.date(), []).append(v)
    days = sorted(by)
    avg = [100 * float(np.mean(by[d])) for d in days]
    ax.plot(days, avg, color=ACCENT, lw=1.4, marker="o", ms=3)
    ax.set_ylim(-5, 105)
    ax.set_ylabel("% fail")
    ax.set_title(f"{model} — linearity fail rate — CURRENT (daily average)")
    fig.tight_layout()
    fig.savefig(OUT / f"B_fail_current_{model}.png", dpi=110)
    plt.close(fig)

    # B2 — proposed: lot fail fractions + n-aware limits
    lots = cluster_lots(samples, use_mean=True)
    closed = [l for l in lots if not l.is_open()]
    if len(closed) < 5:
        return
    base = closed[:-3] if len(closed) > 8 else closed
    pbar = float(np.mean([l.median for l in base]))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    lx = [l.end for l in lots]
    lp = [100 * l.median for l in lots]
    ln = [l.n for l in lots]
    sizes = [max(24, min(180, n * 4)) for n in ln]
    # n-aware 3σ binomial limits around the baseline rate
    for l in lots:
        ucl = pbar + 3 * np.sqrt(max(pbar * (1 - pbar), 1e-6) / max(l.n, 1))
        ax.plot([l.end - timedelta(days=1), l.end + timedelta(days=1)],
                [100 * min(ucl, 1)] * 2, color=RED, lw=1.2, alpha=0.8)
    colors = []
    for l in lots:
        ucl = pbar + 3 * np.sqrt(max(pbar * (1 - pbar), 1e-6) / max(l.n, 1))
        colors.append(AMBER if l.is_open() else (RED if l.median > ucl else ACCENT))
    ax.scatter(lx, lp, s=sizes, c=colors, edgecolors=FG, linewidths=0.6, zorder=5)
    ax.axhline(100 * pbar, color=FG, ls="--", lw=1, alpha=0.8,
               label=f"Baseline lot fail rate {100*pbar:.1f}%")
    ax.set_ylim(-5, 105)
    ax.set_ylabel("% of lot failing")
    ax.set_title(f"{model} — linearity fail rate — PROPOSED per-lot p-chart "
                 f"(size = lot n; red dash = that lot's own 3σ limit)")
    ax.legend(loc="best", fontsize=8, framealpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / f"B_fail_lots_{model}.png", dpi=110)
    plt.close(fig)


def render_distribution(db, model, cutoff):
    rows = db.execute(
        """SELECT t.untrimmed_resistance, t.trimmed_resistance
           FROM track_results t JOIN analysis_results a ON a.id=t.analysis_id
           WHERE a.model=? AND a.file_date>=? AND t.untrimmed_resistance IS NOT NULL
             AND t.untrimmed_resistance < 1e7""", (model, cutoff)).fetchall()
    un = [r[0] for r in rows if r[0] is not None]
    tr = [r[1] for r in rows if r[1] is not None]
    if len(un) < 30:
        return
    lo, hi = np.percentile(un + tr if tr else un, [0.5, 99.5])
    bins = np.linspace(lo, hi, 45)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.hist(un, bins=bins, color=DIM, alpha=0.85, label="Untrimmed (as-fired)")
    if tr:
        ax.hist(tr, bins=bins, color=ACCENT, alpha=0.6, label="After trim")
    ax.set_title(f"{model} — resistance distribution, as-fired vs after trim — "
                 "PROPOSED (the 'raise the as-fired target?' picture)")
    ax.set_xlabel("Ω")
    ax.legend(loc="best", fontsize=8, framealpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / f"C_dist_resistance_{model}.png", dpi=110)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cutoff = (datetime.now() - timedelta(days=WINDOW_DAYS)).isoformat()
    for m in MODELS:
        try:
            render_focus_pair(db, m, cutoff)
            render_fail_pair(db, m, cutoff)
            render_distribution(db, m, cutoff)
            print(f"{m}: done")
        except Exception as e:  # keep going; report
            print(f"{m}: FAILED {type(e).__name__}: {e}")
    print(f"output -> {OUT}")


if __name__ == "__main__":
    main()
