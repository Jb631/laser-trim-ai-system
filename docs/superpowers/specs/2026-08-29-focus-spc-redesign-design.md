# FOCUS / SPC redesign — design spec (2026-08-29)

Approved direction from James (2026-08-29 session), mockups rendered from the
real 8-29 work DB copy (`focus_list_mockup.png`, `focus_chart_no1.png` —
committed in this directory alongside the spec; those two PNGs ARE the archived
approval record, not regenerable output, because the mockup script that drew
them was throwaway and was not kept).

## Problem

James, on the current Triage page (all four confirmed): too noisy / can't
rank; can't tell current from stale; no cost/impact link; doesn't trust the
alerts. Context: the new company president talks SPC — the app should speak
the same language, correctly, while staying readable to a non-SPC-expert.

## Decisions (locked with James)

1. **Membership** ("currently drifting"): a model is on the FOCUS list only
   while an out-of-control signal fires within its **last 5 lots** (closed or
   open). Self-clearing: clean recent lots drop it off. No disposition queue.
2. **Ranking**: **excess failing units per week** = max(p_recent − p̄, 0) ×
   units/week over the model's last 28 days of production. Self-aging: a
   model not run recently decays in volume and sinks; the 60-day
   active cutoff removes it entirely. Biggest bleed first — office bandwidth
   is the constraint, not statistics.
3. **Standard chart**: the **lot p-chart** (fail fraction per production lot,
   n-aware binomial limits around the model's own baseline) becomes the
   app's standard representation for fail-rate data; the lot control chart
   (median metric per lot vs ±2σ/3σ limits) for continuous metrics. The
   unit-dot view remains one click away (toggle) for digging into units.
4. **Headline metric**: linearity fail fraction (the zero-tolerance customer
   disposition). Process metrics (sigma family, resistance, trim effort) and
   FT metrics stay as the drill-down "why" on the Model page — not FOCUS
   membership in v1.
5. **Plain language everywhere**: every flagged point carries a sentence a
   non-SPC reader can act on ("54% of 69 units failed — expected at most
   40%"). James should learn SPC *from* the app.
6. **One computation**: the row, the sparkline, the Model-page chart, and the
   membership decision all read the SAME computed object. The list can never
   claim what the chart doesn't show — this is the trust repair.

## Architecture

### New module: `ml/spc.py` — the single SPC core

Pure functions, no Tk, no globals. Builds on `ml/lots.py` (unchanged lot
clustering — day-cluster, gap > `LOT_GAP_DAYS`=3).

```python
@dataclass
class SpcPoint:            # one lot, one judgment
    start: datetime; end: datetime
    value: float           # fail fraction (p-chart) or lot median (control chart)
    n: int
    ucl: float; lcl: float # n-aware limits for THIS lot
    center: float          # baseline p̄ / mean
    ooc: bool              # beyond limit AND n >= MIN_LOT_N
    is_open: bool
    note: str              # plain-English sentence when ooc, else ""

@dataclass
class SpcSeries:
    model: str; metric: str
    points: list[SpcPoint]         # oldest→newest, window = last 30 lots
    baseline_n_lots: int; baseline_units: int
    chronic: bool                  # baseline itself high but stable (see below)

@dataclass
class FocusEntry:
    model: str
    series: SpcSeries              # the same object the chart draws
    excess_per_week: float
    units_per_week: float
    p_base: float; p_recent: float
    n_flagged_recent: int          # of last RECENT_K lots
    last_lot_end: datetime
    verdict: str                   # the row's plain-English line

def compute_spc_series(db, model, metric, *, respect_requalification=True) -> SpcSeries
def compute_focus_list(db, *, anchor=None) -> list[FocusEntry]   # one bulk pass
```

Rules v1: single rule — point beyond limit (3σ / binomial UCL) with
n ≥ `MIN_LOT_N`=3. Western Electric run rules are a later increment; the
dataclass carries `note`, so adding rules changes no consumer.

Baseline: closed lots older than the newest `RECENT_K`=5, lots with
n ≥ MIN_LOT_N, weighted by n; respects the existing baseline
requalification floor date (same source as drift training). Degenerate
guard: if baseline has < `MIN_LOTS_TRAIN`=8 usable lots → model is "not
judged" (never fake limits).

`compute_focus_list` is ONE bulk query over analysis_results (status,
file_date, model, ~1000-day window) then in-memory clustering — the mockup
did 52 active models this way in ~2 s against the 3.6 GB DB. No per-model
session loops (the N+1 in `get_triage_alerts` does not carry over).

### Chronic flag

`chronic = (p̄ ≥ 0.15)` on a model whose recent lots are NOT out of control.
Control limits describe "its own normal", not "good" — 8340-1 idles at 65%
fail and would otherwise look calm. Chronic models appear in a compact
second strip under FOCUS ("chronically high, stable — different problem:
process capability, not drift"), ranked by fail units/week, capped at 5.

### Triage page (rework in place)

- Zone 1 `FOCUS — drifting now, biggest first`: ranked rows exactly per the
  approved mockup — rank #, model, "failing ~N more units/week than its own
  baseline", sub-line "K of last 5 lots out of control · fail rate a%→b% ·
  ~V units/wk", right-aligned "last lot <date> (Nd ago) · lot still open",
  inline sparkline (last 20 lots, shaded expected band, red flagged recent
  lots, amber older excursions, hollow open lot). Click → Model page with
  the same SpcSeries focused. Cap 7 rows + "N more models with smaller
  signals — click to expand".
- Zone 2: chronic strip (above), then the existing browse list unchanged.
- The header states the anchor: "as of last processed data <date>".
- Refresh: recomputed on page show and after every processing batch (the
  existing after-batch hook already lands there); no manual retrain needed.
- The σ-cards wall, `get_triage_alerts` ordering, and the σ gloss paragraph
  are replaced by this. The multi-metric CUSUM/EWMA detector KEEPS running
  (it feeds the Model page drift table and early-warning pills); it no
  longer decides FOCUS membership.

### Model page

- The model's headline chart becomes the full p-chart per the approved
  mockup: shaded band, baseline line labeled in %, n= labels, flagged recent
  lots annotated with the plain-English sentence, older excursions counted
  in one amber note. Title carries the reading key ("shaded = what this
  model's history says a lot of that size can do by chance").
- Unit-dot view stays as a toggle (existing chart retained behind it).
- Continuous watched metrics reuse the same renderer with lot medians and
  ±2σ/3σ limits (SpcSeries is metric-agnostic).

### Exports

Evidence pack / save-chart use the same SpcSeries (chart parity with the
screen — extends the 2026-07 "exported numbers contradict the screen" fix).
Printable president-ready SPC report = later increment (approach C), not v1.

## Error handling

- Models below MIN_LOTS_TRAIN lots: excluded from FOCUS, shown as "not
  enough lot history to judge" on the Model page chart area — never fake
  limits.
- ERROR/UNTRIMMED statuses excluded from fail series (matches
  `linearity_fail_fraction`).
- Empty focus list renders the existing "all within tolerance" empty state
  with the anchor date.

## Testing

- Unit tests (`tests/test_spc_core.py`): synthetic lot fixtures — limit
  math (known-answer binomial), membership window edges (flag at lot -5 vs
  -6), open-lot handling, baseline exclusion of recent K, requalification
  floor, chronic flag, degenerate baselines, self-aging rank (28-day volume
  decay), verdict wording snapshot.
- Consistency invariant test: FocusEntry.verdict numbers == numbers derived
  from its own SpcSeries (the one-computation guarantee).
- Sweep entries (same commit): triage FOCUS rows == compute_focus_list
  output; Model-page chart flagged count == series ooc count; no weak
  assertions (an ERROR result must FAIL the check).
- chart_qa_render_all gains the p-chart/control-chart variants across the
  existing model matrix; PNGs inspected.

## Out of scope (explicit)

- Western Electric run rules, Cpk-on-row, printable SPC report (increment 2).
- FT-lot FOCUS membership (drill-down only in v1).
- Any change to drift training/advance internals.

## Implementation addenda (2026-08-30)

Recorded after Tasks 1–7 shipped (HEAD `d007381`). Two intentional deviations
from this spec's original wording, plus three implementation details worth
knowing before touching the code.

1. **`compute_focus_list` returns `FocusResult`, not `list[FocusEntry]`.**
   `FocusResult(focus, chronic, anchor)` bundles the ranked list with the
   chronic strip and the anchor date the whole page reads from — the three
   things Triage needs from one call, not three. `ml/spc.py`.
2. **`judged` requires 8 TOTAL lots in the window, not 8 baseline lots.**
   This spec's Architecture section reads "if baseline has < `MIN_LOTS_TRAIN`
   =8 usable lots → not judged"; the shipped rule is `len(lots) <
   MIN_LOTS_TRAIN (8) or len(baseline) < MIN_BASELINE_LOTS (3)` — i.e. 8 lots
   total (baseline + recent K combined), of which at least 3 must be
   baseline. This is the mockup rule James approved live against the work
   DB (resolved in Task 1), not a reinterpretation of the design intent.
   `ml/spc.py`.
3. **The chronic strip renders inside the FOCUS scroll body, not below it.**
   `_chronic_heading` / `_chronic_body` are children of the same
   `CTkScrollableFrame` (`self._body`) the ranked rows use, packed at a fixed
   `BODY_HEIGHT`. Packed on `self` instead, five chronic rows plus their
   heading grew the zone ~214px and starved the browse list beneath it at
   the app's 1400x900 default (Task 5 geometry pass). `focus_list_zone.py`.
4. **Sparklines draw the whole series window, not a trimmed last-20 tail.**
   `_draw_sparkline` plots every point in `entry.series.points` (up to
   `SERIES_WINDOW`=30 lots), because `flag_idx`/`old_idx` are positions into
   that same list — trimming the drawn points without re-deriving the
   indices would silently mark the wrong lots. `focus_list_zone.py`.
5. **Verdict excess formats `.1f` below 10, `.0f` at or above 10.** "failing
   ~N more units/week" reads e.g. `~7.3` under 10 units/week, where a whole
   number would round away the only signal there is, and `~42` at or above
   it, where a decimal would read as false precision. `ml/spc.py`.
