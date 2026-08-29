# App shape + INVESTIGATE page — design spec (2026-08-29)

Companion to `2026-08-29-focus-spc-redesign-design.md`. Approved by James
(2026-08-29): the app reorganizes around his three verbs — INGEST, WATCH,
INVESTIGATE — as three destinations. Derived from his own week: process
folders one at a time, then export a model to Excel just to compute
historical avg/min/max (all units vs lin-passing) for resistance and
electrical angle, then hunt for a unit's chart; the app is used mostly to
investigate models production has already flagged; the chart document and
Excel export are what he shows engineers and his boss.

## The three destinations

### 1. HOME (landing)

- **"Process everything new"** — one button. Runs the remembered folder
  list (each laser folder + the Final Test folder) sequentially through the
  existing batch pipeline, with the per-batch phase line and one combined
  summary ("3 folders · 214 new files · 2 min 40 s"). Folders are
  configured once (add/remove/reorder in Settings; first run prompts).
  Feasible now that a full-share scan is minutes (75c0707).
- Below it: the **FOCUS list** exactly per the FOCUS spec (drifting now,
  biggest first, chronic strip, anchor date in the header).
- The existing Process page's picker remains reachable from HOME
  ("process a specific folder…") for one-off folders — same worker, no
  duplicate pipeline.
- Dashboard: NOT deleted in this phase. Nav de-emphasizes it; its fate
  (fold the used pieces into HOME vs retire) is James's call after HOME
  lands and he's lived with it.

### 2. INVESTIGATE (evolution of the Model page)

The screen that replaces the Excel round-trip. Top-to-bottom:

- **Model search** (existing searchable picker) + **window selector**
  (all history · last 90 days · custom range) + **lot selector** (lots
  from the shared clustering, newest first, "current lot" default when one
  is open).
- **Stats table** (day one, James-selected groups). Rows:
  - Resistance: untrimmed, trimmed (Ω)
  - Electrical angle (measured)
  - Linearity: max error, margin to spec limit
  - Trim behavior: trim pass count, % needing no trim (already met spec
    before trim), sigma gradient
  Columns: n · avg · min · max, split **ALL units | LIN-PASSING units**.
  Lin-passing = overall_status in (PASS, WARNING) — linearity is the
  disposition; WARNING is the sigma watch on accepted units. FAIL = the
  linearity rejects. Same rule as unit-level yield. NULL metric values are
  excluded per-cell with visible n, never imputed.
- **Lot vs history**: when a lot is selected, each stats row gains "this
  lot" values beside the historical band, with a plain-English SPC verdict
  per metric from the shared SPC core (`SpcSeries` on the continuous
  metric): "starting resistance for this lot sits above everything this
  model has done (avg 10.4 kΩ vs normal 8.1–9.6 kΩ)" / "within its
  normal". This is the on-screen answer to "starting resistance is
  different / angle too far out / % of lot didn't trim".
- **Unit finder**: serial search box → opens the existing unit chart modal
  directly (today he has to hunt).
- **Outputs stay first-class** (they are what engineers and the boss see):
  "Save chart…" (existing print-ready document) and "Export Excel" as
  prominent buttons. The Excel pack gains a sheet mirroring the stats
  table + lot verdicts — what he shows people equals what the screen says.
- Existing Model-page content (p-chart per FOCUS spec, drift table,
  smoothness / trim-vs-FT / history tabs) remains below as the deep-dive
  tabs. Nothing currently reachable is lost.

### 3. SETTINGS — unchanged (gains only the ingest folder list).

## Data layer

One bulk query per (model, window) feeding the whole stats table — column
selects over analysis_results + track_results (+ smoothness where needed),
computed in one pass; no per-metric session loops. Lot assignment reuses
`ml/lots.py` clustering via the SPC core. All reads on worker threads via
the existing page-worker pattern; never on the Tk thread.

## Build order

1. **FOCUS** (approved spec) — the WATCH destination's heart.
2. **INVESTIGATE upgrade** — stats table + lot-vs-history + unit search +
   Excel parity sheet.
3. **HOME + shell** — one-click ingest, nav becomes HOME · INVESTIGATE ·
   SETTINGS, Dashboard de-emphasized pending James's retirement call.

Each step lands with tests + sweep entries in the same commit and is
verified against the real 8-29 DB copy before James pulls it at work.

## Testing

- Unit tests: stats-table computation against hand-built fixtures — the
  ALL vs LIN-PASSING split, NULL exclusion with per-cell n, window edges,
  lot-vs-history verdict thresholds (shared SPC core tests already cover
  limit math).
- Sweep entries: stats table vs raw SQL on the live DB (same discipline as
  the dashboard checks); Excel stats sheet == on-screen table values;
  one-click ingest runs the same pipeline as the single-folder path
  (byte-identical batch summaries on test_files).
- chart_qa additions per FOCUS spec cover the chart side.

## Out of scope

- Scheduled/background ingestion outside the app (James runs the .bat on demand
  and closes it — respected).
- Printable president SPC report (increment 2, after FOCUS proves out).
- V5 page retirement and Dashboard removal (explicit later decision).
