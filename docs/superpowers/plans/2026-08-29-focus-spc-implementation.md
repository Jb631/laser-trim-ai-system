# FOCUS / SPC Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Triage σ-card wall with a ranked FOCUS list + SPC lot charts driven by ONE shared computation, per the approved spec.

**Architecture:** New pure module `ml/spc.py` computes lot-based SPC series and the ranked focus list; the Triage page's top zone, the Model page's headline chart, and the evidence export all consume the same objects. The existing CUSUM/EWMA detector keeps running for the Model-page drift table but no longer decides what the landing page shows.

**Tech Stack:** Python 3.14 (`.venv/bin/python`), SQLAlchemy 2.0 ORM, customtkinter + matplotlib (TkAgg embedded), pytest.

**Spec:** `docs/superpowers/specs/2026-08-29-focus-spc-redesign-design.md` (read it first; this plan implements it. Companion context: `docs/superpowers/specs/2026-08-29-app-shape-investigate-design.md`.)

## Global Constraints

- Run everything with `.venv/bin/python` (framework CPython 3.14.2 with working Tk).
- Workers NEVER call Tk; pages load data on a `threading.Thread` and apply via `self.safe_after(...)` (see `gui/v6/pages/triage_page.py:57-63` for the canonical pattern).
- SQLAlchemy 2.0 syntax (`case()` not `func.case()`); column-tuple queries, not full-entity loads, for bulk reads.
- Domain rule: linearity decides pass/fail (FAIL = linearity reject); WARNING = sigma watch on an ACCEPTED unit. Never present sigma as a rejection.
- Weak assertions are forbidden in sweeps/tests: a check that passes on an ERROR result is a bug.
- QA sweeps are mandatory before calling the work done: `scripts/app_qa_sweep.py` and `scripts/chart_qa_render_all.py` (Task 7 makes both runnable against a copy of `8-29-2026_data/data/analysis.db`; NEVER write to that original file, and NEVER create `data/analysis.db` in the repo — a test's skip semantics depend on its absence).
- Commit after each task with a message explaining the WHY (James reads these to learn).
- Plain language in all user-facing copy; σ/SPC terms always accompanied by a domain-words gloss.

## File Structure

- Create: `src/laser_trim_analyzer/ml/spc.py` — dataclasses + series builders + focus-list computation (pure logic; one DB loader function).
- Create: `src/laser_trim_analyzer/gui/v6/widgets/focus_list_zone.py` — FOCUS rows + chronic strip widget.
- Create: `tests/test_spc_core.py`, `tests/test_spc_db.py`.
- Modify: `src/laser_trim_analyzer/gui/v6/widgets/focus_chart.py` (add `set_spc_series`), `src/laser_trim_analyzer/gui/v6/pages/triage_page.py`, `src/laser_trim_analyzer/gui/v6/pages/model_page.py` (`_load_focus_series` ~:355), `src/laser_trim_analyzer/export/evidence.py` (add Lots sheet), `scripts/app_qa_sweep.py`, `scripts/chart_qa_render_all.py`, `tests/test_spec3b_triage.py`, `BRING_TO_WORK.md`.
- Delete (Task 5, after references are gone): `gui/v6/widgets/flagged_cards_zone.py`, `gui/v6/widgets/model_alert_card.py`.

---

### Task 1: SPC core dataclasses + pure series builders

**Files:**
- Create: `src/laser_trim_analyzer/ml/spc.py`
- Test: `tests/test_spc_core.py`

**Interfaces:**
- Consumes: `laser_trim_analyzer.ml.lots.cluster_lots`, `Lot`, `LOT_GAP_DAYS=3`, `MIN_LOT_BASELINE_N=3`, `MIN_LOTS_TRAIN=8` (all exist).
- Produces (later tasks rely on these EXACT names):

```python
RECENT_K = 5            # membership window: last K lots (incl. open)
ACTIVE_DAYS = 60        # focus list requires production this recent
CHRONIC_PBAR = 0.15     # chronic-but-stable threshold (James-approved)
SERIES_WINDOW = 30      # lots carried per series
UNITS_WEEK_DAYS = 28    # volume window for units/week

@dataclass(frozen=True)
class SpcPoint:
    start: datetime; end: datetime
    value: float            # fail fraction (fraction metrics) or lot median
    n: int
    ucl: float; lcl: float; center: float   # nan when series not judged
    ooc: bool               # out of control (see per-builder rule)
    is_open: bool
    note: str               # plain-English sentence when ooc, else ""

@dataclass
class SpcSeries:
    model: str; metric: str
    points: List[SpcPoint]          # oldest→newest, last SERIES_WINDOW lots
    judged: bool                    # False = not enough history; no limits drawn
    p_base: float                   # baseline center (nan when not judged)
    baseline_n_lots: int
    baseline_units: int
    chronic: bool                   # judged, p_base >= CHRONIC_PBAR, no recent ooc

def build_fraction_series(model: str, metric: str,
                          samples: List[Tuple[datetime, float]],  # (date, 0.0/1.0)
                          *, anchor: datetime,
                          requal_floor: Optional[datetime] = None) -> SpcSeries
def build_continuous_series(model: str, metric: str,
                            samples: List[Tuple[datetime, float]],
                            *, anchor: datetime,
                            requal_floor: Optional[datetime] = None) -> SpcSeries
```

**Semantics (locked; resolves the spec's baseline-count ambiguity the way the
James-approved mockup behaved):**
- Drop samples with `date < requal_floor` (when set) or `date > anchor + 1 day` (future-dated file artifacts).
- Lots: `cluster_lots(samples, use_mean=True)` for fraction series, `use_mean=False` for continuous. Keep only the last `SERIES_WINDOW` lots in `points`, but compute the baseline from ALL clustered lots before trimming.
- Open lot: `(anchor - lot.end).days <= LOT_GAP_DAYS`; only the newest lot can be open.
- Recent window = last `RECENT_K` lots (open included). Baseline lots = all lots EXCEPT the recent window, filtered to `n >= MIN_LOT_BASELINE_N`; if that filter empties, fall back to all non-recent lots.
- `judged = (total_lots >= MIN_LOTS_TRAIN) and (len(baseline_lots) >= 3)`. Not judged → every point gets `ucl=lcl=center=float("nan")`, `ooc=False`, `note=""`; `p_base=float("nan")`.
- Fraction: `p_base` = Σ(lot.median·lot.n)/Σ(lot.n) over baseline lots (`.median` holds the MEAN for fraction metrics — see `lots.py` MEAN_AGGREGATED_METRICS). Per-lot `ucl = p_base + 3*sqrt(max(p_base*(1-p_base), 1e-9)/lot.n)`, `lcl = max(p_base - 3*sqrt(...), 0.0)`. `ooc = value > ucl and n >= MIN_LOT_BASELINE_N` (upper side only — a too-GOOD lot is not an alarm). Note when ooc: `f"{value*100:.0f}% of {n} units failed — expected at most {ucl*100:.0f}%"`.
- Continuous: `p_base` (center) = mean of baseline lot medians; `sd` = population std of baseline lot medians. `sd < 1e-12` → `judged=False` (degenerate guard). `ucl/lcl = center ± 3*sd` (same for every point). `ooc = abs(value-center) > 3*sd and n >= MIN_LOT_BASELINE_N` (two-sided). Note when ooc: `f"lot median {value:.4g} — outside this model's normal ({lcl:.4g} to {ucl:.4g})"`.
- `chronic = judged and p_base >= CHRONIC_PBAR and not any(pt.ooc for pt in points[-RECENT_K:])` (fraction builder only; continuous always `chronic=False`).

- [ ] **Step 1: Write the failing tests** — `tests/test_spc_core.py`:

```python
"""SPC core: pure lot-series builders (no DB, no Tk)."""
from datetime import datetime, timedelta

import math
import pytest

from laser_trim_analyzer.ml.spc import (
    ACTIVE_DAYS, CHRONIC_PBAR, RECENT_K, SERIES_WINDOW,
    SpcPoint, SpcSeries, build_continuous_series, build_fraction_series)

D0 = datetime(2026, 1, 5)


def _lot_samples(day: datetime, n: int, fails: int):
    """n units on one day, `fails` of them failing."""
    return [(day, 1.0 if i < fails else 0.0) for i in range(n)]


def _make_history(n_lots=12, n_per=20, fails_per=2, gap=7, start=D0):
    """n_lots clean lots, one per week (gap > LOT_GAP_DAYS => distinct lots)."""
    out = []
    for k in range(n_lots):
        out += _lot_samples(start + timedelta(days=gap * k), n_per, fails_per)
    return out


def test_binomial_limits_known_answer():
    # baseline p = 0.10 across 7 baseline lots of 20; UCL for n=20:
    # 0.10 + 3*sqrt(0.09/20) = 0.5012...  -> a 40% lot stays in control,
    # an 11/20 (55%) lot is out.
    samples = _make_history(n_lots=12, n_per=20, fails_per=2)
    anchor = samples[-1][0]
    s = build_fraction_series("M", "linearity_fail_fraction", samples, anchor=anchor)
    assert s.judged
    assert s.p_base == pytest.approx(0.10)
    last = s.points[-1]
    assert last.ucl == pytest.approx(0.10 + 3 * math.sqrt(0.09 / 20), abs=1e-9)
    assert not last.ooc


def test_ooc_flag_and_note_wording():
    samples = _make_history(n_lots=11, n_per=20, fails_per=2)
    bad_day = samples[-1][0] + timedelta(days=7)
    samples += _lot_samples(bad_day, 20, 11)          # 55% fail lot
    s = build_fraction_series("M", "linearity_fail_fraction", samples, anchor=bad_day)
    last = s.points[-1]
    assert last.ooc
    assert last.note == f"55% of 20 units failed — expected at most {last.ucl*100:.0f}%"


def test_membership_window_edge():
    # An excursion RECENT_K lots ago is still "recent"; one lot older is not.
    base = _make_history(n_lots=10, n_per=20, fails_per=2)
    bad_day = base[-1][0] + timedelta(days=7)
    hist = base + _lot_samples(bad_day, 20, 12)
    # exactly RECENT_K - 1 clean lots after the bad one -> bad lot is points[-RECENT_K]
    for k in range(1, RECENT_K):
        hist += _lot_samples(bad_day + timedelta(days=7 * k), 20, 2)
    anchor = hist[-1][0]
    s = build_fraction_series("M", "m", hist, anchor=anchor)
    assert s.points[-RECENT_K].ooc                      # in the window
    # one more clean lot pushes it out of the recent window
    hist2 = hist + _lot_samples(anchor + timedelta(days=7), 20, 2)
    s2 = build_fraction_series("M", "m", hist2, anchor=hist2[-1][0])
    assert not any(pt.ooc for pt in s2.points[-RECENT_K:])


def test_open_lot_marked_and_baseline_excludes_recent():
    samples = _make_history(n_lots=12, n_per=20, fails_per=2)
    anchor = samples[-1][0] + timedelta(days=1)         # newest lot still open
    s = build_fraction_series("M", "m", samples, anchor=anchor)
    assert s.points[-1].is_open and not any(p.is_open for p in s.points[:-1])
    assert s.baseline_n_lots == 12 - RECENT_K
    assert s.baseline_units == (12 - RECENT_K) * 20


def test_not_judged_below_min_lots():
    samples = _make_history(n_lots=7, n_per=20, fails_per=2)   # < MIN_LOTS_TRAIN
    s = build_fraction_series("M", "m", samples, anchor=samples[-1][0])
    assert not s.judged
    assert math.isnan(s.p_base) and math.isnan(s.points[-1].ucl)
    assert not any(pt.ooc for pt in s.points)


def test_requal_floor_drops_older_lots():
    samples = _make_history(n_lots=14, n_per=20, fails_per=2)
    floor = samples[0][0] + timedelta(days=7 * 4)       # drop first 4 lots
    s = build_fraction_series("M", "m", samples, anchor=samples[-1][0],
                              requal_floor=floor)
    assert len(s.points) == 10


def test_future_dated_samples_excluded():
    samples = _make_history(n_lots=12, n_per=20, fails_per=2)
    anchor = samples[-1][0]
    samples += _lot_samples(anchor + timedelta(days=200), 5, 5)   # file-date junk
    s = build_fraction_series("M", "m", samples, anchor=anchor)
    assert s.points[-1].end <= anchor


def test_chronic_flag():
    hot = _make_history(n_lots=12, n_per=20, fails_per=8)   # stable 40% fail
    s = build_fraction_series("M", "m", hot, anchor=hot[-1][0])
    assert s.judged and s.chronic and not any(p.ooc for p in s.points[-RECENT_K:])
    cool = _make_history(n_lots=12, n_per=20, fails_per=1)  # 5% < CHRONIC_PBAR
    s2 = build_fraction_series("M", "m", cool, anchor=cool[-1][0])
    assert not s2.chronic


def test_series_window_cap():
    samples = _make_history(n_lots=SERIES_WINDOW + 10, n_per=5, fails_per=0)
    s = build_fraction_series("M", "m", samples, anchor=samples[-1][0])
    assert len(s.points) == SERIES_WINDOW
    assert s.baseline_n_lots == SERIES_WINDOW + 10 - RECENT_K  # baseline saw ALL lots


def test_continuous_two_sided_and_degenerate():
    vals = []
    for k in range(12):
        day = D0 + timedelta(days=7 * k)
        vals += [(day, 100.0 + (k % 3)) for _ in range(10)]   # medians 100..102
    jump_day = vals[-1][0] + timedelta(days=7)
    vals += [(jump_day, 140.0) for _ in range(10)]
    s = build_continuous_series("M", "resistance", vals, anchor=jump_day)
    assert s.judged and s.points[-1].ooc
    assert "outside this model's normal" in s.points[-1].note
    flat = [(D0 + timedelta(days=7 * k), 5.0) for k in range(12) for _ in range(10)]
    s2 = build_continuous_series("M", "r", flat, anchor=flat[-1][0])
    assert not s2.judged                                    # zero spread guard
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_spc_core.py -q -p no:warnings`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'laser_trim_analyzer.ml.spc'`

- [ ] **Step 3: Implement `ml/spc.py`** (dataclasses + both builders exactly per the Semantics block above; ~150 lines. Module docstring: why one shared computation exists — "the list can never claim what the chart doesn't show". Import lots helpers; no DB imports at module top except in the Task 2 functions.)

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_spc_core.py -q -p no:warnings`
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/ml/spc.py tests/test_spc_core.py
git commit -m "SPC core: lot-series builders with n-aware limits + plain-English verdicts

One shared computation for list, chart, and export (FOCUS spec 2026-08-29):
p-chart limits per lot size, membership = OOC within last 5 lots, chronic-
but-stable detection, honest 'not judged' below 8 lots."
```

---

### Task 2: DB loaders — `compute_spc_series` + bulk `compute_focus_list`

**Files:**
- Modify: `src/laser_trim_analyzer/ml/spc.py` (append)
- Test: `tests/test_spc_db.py`

**Interfaces:**
- Consumes: Task 1 builders; `DatabaseManager.get_baseline_requalification(model)` → `(effective_date_str, note, set_at) | None` (manager.py:344); `ml.drift_training._load_samples_with_dates(db, model, metric)` → `[(file_date, value, row_id)]` for single-model series.
- Produces:

```python
@dataclass
class FocusEntry:
    model: str
    series: SpcSeries
    excess_per_week: float
    units_per_week: float
    p_base: float; p_recent: float
    n_flagged_recent: int
    last_lot_end: datetime
    verdict: str        # "failing ~7 more units/week than its own baseline"
    sub_line: str       # "2 of last 5 lots out of control · fail rate 25% → 51% · ~27 units/wk"

@dataclass
class FocusResult:
    focus: List[FocusEntry]      # ranked, excess_per_week desc
    chronic: List[FocusEntry]    # ranked by p_base*units_per_week desc, max 5
    anchor: Optional[datetime]   # None = empty database

def compute_spc_series(db, model: str, metric: str = "linearity_fail_fraction",
                       *, anchor: Optional[datetime] = None) -> SpcSeries
def compute_focus_list(db, *, anchor: Optional[datetime] = None) -> FocusResult
```

**Semantics:**
- `compute_spc_series`: samples via `_load_samples_with_dates` (drops None values); `requal_floor` from `get_baseline_requalification` (parse `effective_date` with `datetime.fromisoformat`); anchor default = newest sample date ≤ now+1d; fraction vs continuous builder chosen by `metric in laser_trim_analyzer.ml.drift_types.FRACTION_METRICS`.
- `compute_focus_list`: ONE column query (mirror `_load_samples_with_dates`'s
  linearity branch, but ungrouped by model):

```python
with db.session() as s:
    rows = (s.query(DBAR.model, DBAR.file_date, DBAR.overall_status)
            .filter(DBAR.overall_status.in_([StatusType.PASS, StatusType.WARNING,
                                             StatusType.FAIL]),
                    DBAR.file_date.isnot(None), DBAR.model.isnot(None))
            .order_by(DBAR.model, DBAR.file_date).all())
```

  anchor = max(file_date ≤ now+1d) (None → `FocusResult([], [], None)`). Per model: skip if last sample older than `ACTIVE_DAYS` before anchor; requal floor per model; `build_fraction_series`; skip `judged=False`. `units_per_week = len(samples within UNITS_WEEK_DAYS of anchor) / (UNITS_WEEK_DAYS/7)`. Flagged recent lots = `[pt for pt in points[-RECENT_K:] if pt.ooc]`; if any → FocusEntry with `p_recent` = Σ(value·n)/Σ(n) over flagged lots, `excess_per_week = max(p_recent - p_base, 0) * units_per_week`, verdict strings EXACTLY:
  `verdict = f"failing ~{excess_per_week:.0f} more units/week than its own baseline"`
  `sub_line = f"{n_flagged_recent} of last {RECENT_K} lots out of control · fail rate {p_base*100:.0f}% → {p_recent*100:.0f}% · ~{units_per_week:.0f} units/wk"`
  elif `series.chronic` → chronic list, `verdict = f"runs ~{p_base*100:.0f}% fail, stable — capability problem, not drift"`, `sub_line = f"~{units_per_week:.0f} units/wk"`, `excess_per_week = 0.0`, `p_recent = p_base`, `n_flagged_recent = 0`.
  Sort focus by `excess_per_week` desc; chronic by `p_base*units_per_week` desc capped at 5.

- [ ] **Step 1: Write the failing tests** — `tests/test_spc_db.py` (build a tmp DB exactly like `tests/test_ft_stat_fastpath.py` does — `DatabaseManager(tmp_path / "t.db")` — and insert `AnalysisResult` rows with only the fields the query touches: model, serial, file_date, overall_status, filename):

```python
"""SPC DB layer: compute_spc_series / compute_focus_list against a tmp DB."""
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.database.models import AnalysisResult, StatusType
from laser_trim_analyzer.ml.spc import (
    RECENT_K, compute_focus_list, compute_spc_series)

D0 = datetime(2026, 1, 5)


@pytest.fixture()
def db(tmp_path):
    return DatabaseManager(tmp_path / "t.db")


def _add_lot(db, model, day, n, fails):
    with db.session() as s:
        for i in range(n):
            s.add(AnalysisResult(
                model=model, serial=f"{model}-{day:%m%d}-{i}",
                filename=f"{model}_{i}_{day:%m-%d-%Y}.xls", file_date=day,
                overall_status=StatusType.FAIL if i < fails else StatusType.PASS))


def _seed(db, model, n_lots=12, fails_last=0, start=D0, n_per=20, base_fails=2):
    for k in range(n_lots - 1):
        _add_lot(db, model, start + timedelta(days=7 * k), n_per, base_fails)
    last_day = start + timedelta(days=7 * (n_lots - 1))
    _add_lot(db, model, last_day, n_per, fails_last if fails_last else base_fails)
    return last_day


def test_focus_membership_and_ranking(db):
    d_hot = _seed(db, "HOT", fails_last=12)     # drifting, 20 u/lot
    _seed(db, "CALM", fails_last=2, start=D0 + timedelta(days=1))
    res = compute_focus_list(db)
    assert res.anchor == d_hot
    assert [e.model for e in res.focus] == ["HOT"]
    e = res.focus[0]
    assert e.n_flagged_recent == 1 and e.last_lot_end == d_hot
    assert e.verdict.startswith("failing ~") and "units/week" in e.verdict
    assert f"1 of last {RECENT_K} lots out of control" in e.sub_line


def test_verdict_numbers_match_series(db):
    """The one-computation guarantee: row numbers derive from its own series."""
    _seed(db, "HOT", fails_last=12)
    e = compute_focus_list(db).focus[0]
    flagged = [p for p in e.series.points[-RECENT_K:] if p.ooc]
    p_recent = sum(p.value * p.n for p in flagged) / sum(p.n for p in flagged)
    assert e.p_recent == pytest.approx(p_recent)
    assert e.excess_per_week == pytest.approx(
        max(e.p_recent - e.p_base, 0.0) * e.units_per_week)


def test_inactive_model_leaves_the_list(db):
    _seed(db, "OLD", fails_last=12)                          # ends ~11 weeks in
    _seed(db, "NOW", start=D0 + timedelta(days=200))         # anchor mover
    res = compute_focus_list(db)
    assert all(e.model != "OLD" for e in res.focus)          # > ACTIVE_DAYS stale


def test_chronic_strip(db):
    _seed(db, "SICK", n_lots=12, base_fails=8, fails_last=8)  # stable 40%
    res = compute_focus_list(db)
    assert [e.model for e in res.chronic] == ["SICK"]
    assert not res.focus
    assert "capability problem, not drift" in res.chronic[0].verdict


def test_requalification_floor_respected(db):
    d = _seed(db, "REQ", n_lots=14, fails_last=2)
    db.set_baseline_requalification("REQ", (D0 + timedelta(days=7 * 4)).isoformat())
    s = compute_spc_series(db, "REQ")
    assert len(s.points) == 10


def test_empty_db(db):
    res = compute_focus_list(db)
    assert res.anchor is None and res.focus == [] and res.chronic == []
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_spc_db.py -q -p no:warnings` → `ImportError: cannot import name 'compute_focus_list'`
- [ ] **Step 3: Implement** the two functions + dataclasses per Semantics (append to `ml/spc.py`). Bulk path groups rows by model in one pass (`itertools.groupby` over the ordered query), never a per-model session.
- [ ] **Step 4: Run to verify pass** — expected `6 passed`; also re-run `tests/test_spc_core.py` (still `11 passed`).
- [ ] **Step 5: Commit** — `git add src/laser_trim_analyzer/ml/spc.py tests/test_spc_db.py && git commit -m "SPC DB layer: one bulk pass computes the ranked FOCUS list ..."`

---

### Task 3: p-chart rendering — `FocusChart.set_spc_series`

**Files:**
- Modify: `src/laser_trim_analyzer/gui/v6/widgets/focus_chart.py`
- Test: `tests/test_spc_chart_helpers.py` (new, pure helpers only)

**Interfaces:**
- Consumes: `SpcSeries`/`SpcPoint` from Task 1.
- Produces: `FocusChart.set_spc_series(series: SpcSeries, *, focus_recent: int = RECENT_K) -> None` and module-level pure helpers:

```python
def spc_draw_params(series, focus_recent=RECENT_K) -> dict
    # {"xs": [...], "values": [...], "ucls": [...], "center": float|nan,
    #  "flag_idx": [i...],        # ooc within the recent window -> red + annotated
    #  "old_ooc_count": int,      # ooc outside it -> amber note line
    #  "open_idx": int|None, "labels": {i: note}, "n_labels": ["n=90", ...],
    #  "x_dates": ["07/25", ...], "judged": bool}
```

- [ ] **Step 1: Failing tests** for `spc_draw_params` (no Tk needed): flagged-recent indices vs older ooc split; `labels` only for recent flags; `judged=False` → empty flags and no labels; open lot index. Build series with Task 1 builders (reuse `_make_history` shape inline — copy the helper, do not import from another test file).
- [ ] **Step 2: Run** → fails (`ImportError: spc_draw_params`).
- [ ] **Step 3: Implement.** `spc_draw_params` is pure; `set_spc_series` clears the axes and draws, matching the approved mockup (`docs/` mockup PNGs referenced in the spec): step-filled band 0→UCL per lot (`fill_between(step="mid")`, theme band color `#2c3540` equivalent from ThemeManager — use `t.CARD`-adjacent tone already used by `set_series` for its bands; reuse whatever band color `set_series` uses today for visual continuity), dashed center line labeled `baseline {center*100:.0f}%` (fraction) or `baseline {center:.4g}` (continuous), grey dots, red dots + annotation text for `flag_idx` (alternate `xytext=(-130, 26)` / `(12, -42)` by parity of the index to avoid collisions), hollow marker for the open lot, `n=` labels under the axis at fontsize 7, amber note `f"{old_ooc_count} earlier out-of-control lots in this window (unlabeled)"` top-left when nonzero, title carrying the plain-language key: `f"{model} — {metric_label(metric)} by production lot\nshaded = what this model's history says a lot of that size can do by chance · red = beyond it: something changed"`. `judged=False` → dots only + centered caption "not enough lot history to judge (needs 8 lots)". Call `self.canvas.draw_idle()` at the end (same as `set_series`).
- [ ] **Step 4: Run helper tests** → pass; `.venv/bin/python -c "from laser_trim_analyzer.gui.v6.widgets.focus_chart import spc_draw_params"` imports clean.
- [ ] **Step 5: Commit.**

---

### Task 4: FOCUS list widget

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/focus_list_zone.py`
- Test: extend `tests/test_spc_chart_helpers.py` with the zone's pure row-model helper

**Interfaces:**
- Consumes: `FocusResult`/`FocusEntry` (Task 2), `spc_draw_params` (Task 3), ThemeManager tokens (`CARD`, `TEXT_PRIMARY`, `TEXT_SECONDARY`, `ACCENT`, `SPACE_*`, `font()`), pattern reference: `flagged_cards_zone.py` (scroll-bounded region, `_rendered` bookkeeping, empty state with `last_processed`).
- Produces: `FocusListZone(ctk.CTkFrame)` with `__init__(master, theme, on_row_click: Callable[[str, str], None])` and `set_result(result: FocusResult, last_processed: Optional[datetime] = None) -> None`; module helper `focus_row_texts(entry, anchor) -> dict` (`{"title", "when", "line1", "line2"}` — `when = "last lot {end:%b %d} ({days}d ago)"` with "today" for 0, `" · lot still open"` appended when `series.points[-1].is_open`).

- [ ] **Step 1: Failing test** for `focus_row_texts` (title/when/open-lot suffix/line wording from a synthetic FocusEntry).
- [ ] **Step 2: Run** → ImportError.
- [ ] **Step 3: Implement the zone.** Layout per approved mockup: heading `FOCUS — drifting now, biggest first (N)`; caption line `as of last processed data {anchor:%b %d} · a model is here only while its last 5 lots include one outside its own control limits · ranked by extra failing units per week`; scrollable body height 320; each row = CTkFrame(CARD) with rank number, bold model, right-aligned `when`, `verdict` line, `sub_line` in TEXT_SECONDARY, and a 260×64 px sparkline (matplotlib Figure + FigureCanvasTkAgg, dpi 96) drawn from `spc_draw_params` (band + dots only — no annotations at sparkline size, red recent flags ms=5, amber older ms=3.5, hollow open lot); whole row click-bound (`bind` on the frame AND its labels) → `on_row_click(entry.model, entry.series.metric)`. First 7 rows shown; if more, a text button `+ N more models with smaller signals — show all` toggles the rest (same widget, no pagination). Chronic strip below: heading `CHRONICALLY HIGH — stable, different problem (M)`, one-line rows (model · verdict · sub_line), clickable the same way, no sparkline. Empty state: `All models within tolerance — last processed {when}.` (same copy as today's FlaggedCardsZone). Sparkline canvases must be destroyed with the rows (`_rendered` list pattern + explicit `canvas.get_tk_widget().destroy()`).
- [ ] **Step 4: Import check + helper tests pass.** `.venv/bin/python -c "import laser_trim_analyzer.gui.v6.widgets.focus_list_zone"`.
- [ ] **Step 5: Commit.**

---

### Task 5: Rewire the Triage page; retire the card wall

**Files:**
- Modify: `src/laser_trim_analyzer/gui/v6/pages/triage_page.py`
- Modify: `tests/test_spec3b_triage.py` (read it first; keep its structure)
- Delete: `src/laser_trim_analyzer/gui/v6/widgets/flagged_cards_zone.py`, `src/laser_trim_analyzer/gui/v6/widgets/model_alert_card.py`
- Modify: `src/laser_trim_analyzer/ml/manager.py` — ONLY to update the docstring of `get_triage_alerts` noting the v6 Triage page no longer consumes it (it remains for the drift table / v5 paths; do not delete).

**Interfaces:**
- Consumes: `compute_focus_list` (Task 2), `FocusListZone` (Task 4); keeps `list_known_models`, `active_model_set`, `BrowseZone` as-is.
- Produces: Triage page behavior later tasks/sweeps rely on: `TriagePage._query()` returns `(FocusResult, models, active, last)`; `reload_now()` stays the synchronous test path; row click routes `self.app.set_model_route(model, metric)` + `self.app.show_page("model")`.

- [ ] **Step 1: Update tests first.** Rewrite `tests/test_spec3b_triage.py` assertions that exercise FlaggedCardsZone/get_triage_alerts into: seed tmp DB (reuse `_seed` shape from `tests/test_spc_db.py`, copied inline), instantiate the page under Tk (the file already creates pages — follow its existing harness), `reload_now()`, assert the zone heading shows `(1)`, the drifting model's row exists, clicking (invoke the bound callback directly) routes to the model page with metric `linearity_fail_fraction`, and the "All models" scope toggle still filters ONLY the browse list (FOCUS list content identical in both scopes — assert row count unchanged after `_on_scope_change("All models")`).
- [ ] **Step 2: Run** → fails (page still builds FlaggedCardsZone).
- [ ] **Step 3: Rework the page.** `build_content`: zone header becomes `"WHAT THE APP IS TELLING YOU"` / sub `"drifting now, biggest first — one verdict per lot, self-clearing"`; `FocusListZone` replaces FlaggedCardsZone; `_query` calls `compute_focus_list(self.app.db)` instead of `get_triage_alerts`; `_apply` no longer filters/sorts flagged entries (the compute owns membership + order) — active-scope filtering applies to the browse list only. Delete the two dead widget files; grep repo for lingering imports.
- [ ] **Step 4: Run** `tests/test_spec3b_triage.py`, `tests/test_spc_db.py`, full suite: `.venv/bin/python -m pytest tests/ -p no:warnings 2>&1 | tail -1` — expected `... 1 failed` ONLY if `test_chart_ylim_excludes_outlier_at_zero_volts` still trips its known env issue; anything else = fix before proceeding.
- [ ] **Step 5: Commit.**

---

### Task 6: Model page — SPC headline chart + Lots/Units toggle

**Files:**
- Modify: `src/laser_trim_analyzer/gui/v6/pages/model_page.py` (`_load_focus_series` ~:355, chart area in `build_content` ~:101, `_reload` worker ~:216)
- Test: extend `tests/test_spec3b_triage.py`'s harness file or the model-page test file if one exists (`/usr/bin/grep -rl "ModelPage" tests/` first; follow what exists)

**Interfaces:**
- Consumes: `compute_spc_series` (Task 2), `FocusChart.set_spc_series` (Task 3), `FRACTION_METRICS` from `ml.drift_types`.
- Produces: a `view` state later consumed by exports: `ModelPage._chart_view` in `{"lots", "units"}`; segmented button labeled `Lots · SPC | Units` next to the chart; default `lots` for ALL metrics; `Units` renders the existing `set_series` path unchanged.

- [ ] **Step 1: Failing test** — model page harness: seed tmp DB with a drifting model, route to it, `reload_now()`-equivalent, assert the chart widget received an SpcSeries (spy: monkeypatch `FocusChart.set_spc_series` to record calls) with `metric == "linearity_fail_fraction"`; toggle to Units → `set_series` called once.
- [ ] **Step 2: Run** → fails.
- [ ] **Step 3: Implement.** In the `_reload` worker, build the series off-thread (`compute_spc_series(db, model, metric)`) and pass it through the existing apply path; `_load_focus_series` keeps producing the units series for the Units view. Toggle is a `CTkSegmentedButton` in the chart card header following the Triage scope-toggle styling (`triage_page.py:31-38`). Continuous focus metrics route through `build_continuous_series` automatically via `compute_spc_series`.
- [ ] **Step 4: Run tests + full suite** (same expectation as Task 5).
- [ ] **Step 5: Commit.**

---

### Task 7: Evidence export Lots sheet + sweep/chart-QA coverage + DB-path args

**Files:**
- Modify: `src/laser_trim_analyzer/export/evidence.py` (`export_evidence_pack` :100 — add a sheet), `scripts/app_qa_sweep.py`, `scripts/chart_qa_render_all.py`, `tests/test_spc_db.py` (one export test)
- Sweep DB override: both scripts accept an optional DB path argv (default `data/analysis.db`) so they run on this Mac against a COPY of `8-29-2026_data/data/analysis.db`.

**Interfaces:**
- Consumes: `compute_spc_series`, `compute_focus_list`, `spc_draw_params`; sweep helpers `check(name, ok, detail="")` / `warn(...)` (app_qa_sweep.py:65-70); chart-QA `_save(obj, path, manifest, note)` + `VARIANTS` pattern.
- Produces: evidence pack sheet `"Lots (SPC)"` with columns `lot_start, lot_end, units, fail_rate, expected_max (UCL), out_of_control, open_lot, note`.

- [ ] **Step 1: Failing export test** in `tests/test_spc_db.py`: seed drifting model, `export_evidence_pack(db, "HOT", tmp_path/"e.xlsx")`, read back with pandas: sheet exists, its `out_of_control` flags equal the series' `[p.ooc for p in points]`, its fail_rate column equals `[p.value ...]`.
- [ ] **Step 2: Run** → fails (no such sheet).
- [ ] **Step 3: Implement** the sheet from `compute_spc_series(db, model)` inside `export_evidence_pack` (rows straight from `SpcPoint`s; the same object the screen draws — say so in a comment). Then sweep work in `scripts/app_qa_sweep.py`:
  - `main()` takes `db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO/"data"/"analysis.db"` (both the DatabaseManager and the raw sqlite connect use it).
  - REPLACE the section-3 triage checks that assert `get_triage_alerts` invariants with FOCUS invariants: (a) `compute_focus_list` twice → identical model orderings (determinism); (b) every focus entry has ≥1 ooc point in `points[-RECENT_K:]` and every chronic entry has 0; (c) focus sorted by `excess_per_week` desc; (d) each entry's `p_recent/excess` recompute from its own series within 1e-9 (the one-computation guarantee, same math as `test_verdict_numbers_match_series`); (e) every focus/chronic model exists in `analysis_results` (this is the invariant whose `get_triage_alerts` version was failing on the 8-29 DB — determine whether the new path passes it, and record the outcome in the commit message).
  - `scripts/chart_qa_render_all.py`: add a `_spc()` factory beside `_focus()` (:82) rendering `FocusChart.set_spc_series(compute_spc_series(db, m))` for each `m` in `VARIANTS`, saved as `spc_pchart_<model>.png` via `_save`; accept the same optional DB argv.
- [ ] **Step 4: Verify.** `cp 8-29-2026_data/data/analysis.db "$SCRATCH/qa_copy.db"` (scratchpad); run `.venv/bin/python scripts/app_qa_sweep.py "$SCRATCH/qa_copy.db"` → read every FOCUS-section line; expected 0 new FAILs (pre-existing failures elsewhere: report, don't mask). Run `.venv/bin/python scripts/chart_qa_render_all.py <outdir> "$SCRATCH/qa_copy.db"` and OPEN the `spc_pchart_*.png` files — visually confirm band/dots/annotations render (this is the mandated PNG inspection).
- [ ] **Step 5: Commit.**

---

### Task 8: Docs + final verification

**Files:**
- Modify: `BRING_TO_WORK.md` (new dated section at top), `docs/superpowers/specs/2026-08-29-focus-spc-redesign-design.md` (one-line addendum noting `compute_focus_list` returns `FocusResult`, and the judged-rule resolution from Task 1)

- [ ] **Step 1: BRING_TO_WORK section** — plain words: Triage's card wall is now FOCUS ("drifting now, biggest first, in units per week — a model drops off by itself once its lots run clean"); the Model page headline chart is the lot chart with a Lots/Units toggle; the evidence Excel gains a "Lots (SPC)" sheet matching the screen; nothing to retrain or click after pulling.
- [ ] **Step 2: Full gate, read every line:**
  - `.venv/bin/python -m pytest tests/ -p no:warnings 2>&1 | tail -1` → only the known env failure at most.
  - Sweep + chart QA against the scratch copy (as Task 7) → FOCUS sections green, PNGs inspected.
  - `.venv/bin/python -m src --v6` smoke launch is JAMES's step — do not attempt in an agent (GUI); note it for him instead.
- [ ] **Step 3: Commit + push** `git push origin V6 && git push origin V6:main`.

## Self-Review (done at planning time)

- Spec coverage: membership/ranking/self-aging (T2), chronic strip (T2/T4), standard chart + toggle (T3/T6), one-computation guarantee (T2 test + T7 sweep d), plain-language notes (T1), not-judged honesty (T1/T3), page rework + card retirement (T5), export parity (T7), sweep/chart-QA entries incl. weak-assertion rule (T7), update cadence (existing after-batch hook + on_show — no new work needed; the page recomputes on show, and processing already lands on Triage via `_goto_triage`).
- Deviations from spec, both intentional: `compute_focus_list` returns `FocusResult` (focus+chronic+anchor) rather than a bare list; `judged` requires 8 TOTAL lots (the mockup rule James approved) rather than 8 baseline lots. Task 8 records both in the spec addendum.
- Type consistency: `FocusEntry.series.metric` feeds the row-click route (Task 5) and `set_model_route(model, metric)`; `spc_draw_params` signature identical in Tasks 3/4/7.
