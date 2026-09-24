# Parse upgrades and fixes (2026-09-23)

James, 2026-09-23 (`/goal`): "finish the redesign, complete all the parse upgrades and fixes &
finish the task list" — and treat that as the directive, without pausing to ask. So every
decision below is a **ruling**, written as *what — why — what it costs if wrong*, for James to
overturn at leisure. Nothing here changes what the laser or the final-test station measured; it
changes what the app stores, how it reads what it stored, and what it says about it.

Every number in this document was measured on the rebuilt work database (6.23 GB, opened
read-only) or on the local file corpus. Evidence: `research-error-reasons.md` and
`research-lts-unread-sheets.md` (session scratchpad, 2026-09-23).

---

## 1. A record that failed processing is never a measurement

**Found:** `core/analyzer.py::_create_failed_track` (a track with fewer than 10 data points)
stores `status=ERROR` **and `linearity_pass=False`**. Four Findings analyzers
(`limit_tables`, `cut_setting`, `recipe_change`, `ink_target`) count every track whose
`linearity_pass is not None`, so these ERROR tracks are counted as **linearity failures**:

| model | laser | ERROR tracks counted as FAIL | graded | pass rate as counted | without them |
|---|---|---|---|---|---|
| 8856 | Laser 2 (DLTS) | 75 | 173 | 27.7% | **49.0%** |
| 8856-1 | Laser 2 (DLTS) | 18 | 86 | 57.0% | **72.1%** |
| 8914 | Laser 2 (DLTS) | 1 | 35 | 8.6% | 8.8% |

This breaks two CLAUDE.md rules at once: "a record that failed processing is not a
measurement" and "a failure must never look like a result".

**Ruling 1a:** `_create_failed_track` stores `linearity_pass=None` and `sigma_pass=None` — the
same call `enforce_measurement_backed_verdict` already makes ("the verdict is withdrawn, never
inverted") — why: an ungraded track must not carry a verdict — cost if wrong: none; nothing
graded it.

**Ruling 1b:** `findings/data.py::load_model_tracks` drops failed-processing rows in BOTH of
its queries, using `core/model_stats.failed_processing_statuses()` (the one definition), and
the pass query gets the track query's `system IN ('A','B','C')` filter — why: the 94 rows
already stored with `linearity_pass=0` must stop counting today, without a reprocess — cost if
wrong: none.

## 2. V5's "Apply ML" must not rewrite verdicts

**Found (latent):** `ml/manager.py::apply_to_database` recomputes every track's status with a
sigma-coupled rule — both pass → PASS, both fail → FAIL, anything else → WARNING. The ingest
rule (`core/analyzer.py`, "Determine overall status") is: unusable spec → ERROR; linearity
fails → **FAIL**; sigma passes → PASS; otherwise WARNING. It rewrites every track of every
model with a trained threshold, so with every model trained one click of V5 Settings → Apply
would turn up to **1,701 linearity FAILs into WARNING** (linearity is zero-tolerance), take all
235 ERRORs out of ERROR (141 to WARNING, 94 to FAIL), and turn **6,973 UNTRIMMED sweeps into
WARNING**. It has not been run on the rebuild (no `WARNING` row has `linearity_pass=0`). Only V5
calls it; V6 never does.

**Ruling 2:** the bulk update skips ERROR and UNTRIMMED tracks and uses the ingest rule; the
per-file roll-up follows `processor._determine_overall_status` (all-UNTRIMMED → UNTRIMMED;
UNTRIMMED tracks ignored when judging; any ERROR → ERROR; any FAIL → FAIL; all PASS → PASS;
else WARNING) — why: ML thresholds may move sigma, never linearity — cost if wrong: none.

## 3. Every ERROR says why

**Found:** none of the 237 ERROR rows on the rebuild carries a reason the app can show.
`analysis_results` has no reason column; `processed_files.error_message` is written only on a
separate failure-marker row. The reasons exist, scattered: 140 in
`track_results.linearity_spec_warning`, 94 in `track_results.anomaly_reason`, 3 in the marker
row ("No valid track data found").

**Ruling 3a:** one nullable column, `analysis_results.error_reason` (Text), added by the
existing start-up migration. The processor fills it for every ERROR result — from the ERROR
tracks' `linearity_spec_warning` / `anomaly_reason`, or the file-level error message —
and the manager writes it on insert AND on re-process (`_update_existing_analysis`), and puts
the same text in `processed_files.error_message` on the LINKED row — why: one place to ask
"why is this ERROR?" — cost if wrong: one unused column.

**Ruling 3b:** `enforce_measurement_backed_verdict` records its reason on the track
(`linearity_spec_warning`, only when empty), so it flows into 3a like the analyzer's — why: it
is the same kind of verdict withdrawal — cost if wrong: none.

**Ruling 3c:** the 237 existing rows are NOT back-filled. The Model page reads
`COALESCE(error_reason, linearity_spec_warning, anomaly_reason)`, which explains 234 of them
today; the remaining 3 get their reason when next re-read — why: no write to the production
database for a display string — cost if wrong: 3 rows show no reason until reprocessed.

**Ruling 3d:** the reason is shown in the unit list's *Linearity error* cell, which for an
ERROR row is empty today ("—") — "not graded: insufficient data points" is exactly why that
cell has no value — why: the smallest change that puts the answer where the question is;
the Model page's own relayout (facelift step 2) can move it — cost if wrong: a long reason is
cut short in a narrow cell.

The failure-marker behaviour (which files are retried) is unchanged: the marker still reads
the file-level `errors`, exactly as today.

## 4. Laser 1's `TrimVolts` sheets (TRACKER B1a)

**Found** (4,972 local laser-1 files, 0 read errors, 32 models):

- `TrimVolts N` exists if and only if `Trim N` exists — 4,540 files, zero exceptions. The 432
  files without one are 431 no-cut templates and 1 touch-up file.
- Column *k* is the position at row (*start* + *k*) of `Trim N`, where *start* is the file's
  `Points From Start` when it names both `Points From Start` and `Points From End` ("how many
  points from start to begin reading/trimming"), else its `Initial Points Ignored`.
  *Corrected 2026-09-24 (Task 4 review):* this bullet first said `Initial Points Ignored`
  alone. Held against the machine's own placement (its `VOLTAGES` sheet puts column *k*'s
  last reading at row *start* + *k*, exactly), that placed 6,227 of 6,263 local passes; the
  36 misses are 8340-1 files with `Initial Points Ignored` 1 and `Points From Start` 2, every
  curve one position off. The corrected rule places 6,263 of 6,263. r alone could not see it:
  a straight ramp correlates with any shift of itself (reversed order is still refuted, at
  r = −0.9999). On the work database 4,311 laser-1 files, across many models, carry a Points
  From Start/End that differs from the ignored counts. Each row is one more laser increment;
  each cell is the output voltage after it. Row 0 is non-zero in every engaged column — **the
  only per-position pre-cut reading a laser-1 file carries**. Zeros are end padding only (no
  zero between two readings in any of 107 passes).
- The column count is `Number of Readings (Lin)` − *start* − *end* + 1 (the same pair of
  fields), exact on 6,198 of 6,263 local sheets; the rest are all narrower, 11 of them at the
  old .xls **256-column limit, cut short**. (The first version, with the ignored counts only,
  also called a touch-up's full 120-column sheet truncated against a window of 122.)
- The older template of 6607 and 8232-1 labels the same slot `Start Point` / `End point`
  ("points from start for reading/measuring" — not trimming). It equals the ignored counts on
  every local file, so no local workbook can say which one the machine follows when they
  differ; they do on 1,223 old (2011–2016) files on the work database, holding 14 stored
  `Trim N` passes. Not used — an open question for James, not a ruling.
- The last reading is **not** `Trim N`'s measured value (ratio 0.94–1.23, smooth within a file):
  it is the live reading during the cut, `Trim N` is the verification sweep after it. They must
  never be used interchangeably.
- `VOLTAGES` and `ERRORS` repeat data already read (`VOLTAGES` column *k* ≥ 1 is `TrimVolts k`'s
  last reading). `TRIMDATA` is small and real, but its formula could not be reproduced.
- Laser 1 has 41,174 real trim passes stored today (plus 43,926 `Lin Error` pseudo-passes, which
  have no TrimVolts).

**Ruling 4a:** capture `TrimVolts` only. Not `VOLTAGES`/`ERRORS` (duplicates), not `TRIMDATA` (an
unexplained number invites misuse) — cost if wrong: TRIMDATA needs a second capture later.

**Ruling 4b:** store it on the matching `trim_passes` row, in the shape laser 2's per-position
data already uses: `increment_volts` (JSON; one list per engaged position, zero padding dropped),
`increment_volts_first_row` (the position index of column 0), `increment_volts_truncated`
(true when the sheet hit 256 columns or has fewer columns than the window) — why: one pattern
for all per-position data; the name keeps it apart from `Trim Parameters`' `Trim Volts`, a
setting already stored as `trim_voltage` — cost if wrong: roughly 0.3–0.7 GB more than a
compressed encoding would take.

**Ruling 4c:** existing rows are filled by a resumable back-fill script James runs at work
(`scripts/backfill_increment_volts.py`: laser-1 files only, reads only the `TrimVolts` sheets,
updates only those columns, skips passes already filled, snapshot first), not by a full
reprocess — why: the last full rebuild took about three days; this reads 27% of the files and
a fraction of each — cost if wrong: a night of the work PC's time.

## 5. Three small ones

**Ruling 5a — the database path in `config.yaml` travels.** `Config.save()` writes the absolute
path it resolved at run time (`C:\dev\…\data\analysis.db`); carried to the Mac inside `data/`,
that string is read as a RELATIVE POSIX path and a junk file is created under the repo root.
Save the path relative to the app folder when it lies inside it; resolve relative paths against
the app folder on load; an absolute path written by the other operating system falls back to
the default with a WARNING — cost if wrong: a user who deliberately keeps the database outside
the app folder on one OS must set it again on the other.

**Ruling 5b — `Unknown` is not a model.** 381 analyses carry the parser's `"Unknown"` sentinel
(test files such as `…_spike_…` and `BLUE TEST EVERY HALF DEGREE…`). `get_known_models()`
returned it as a known model; it no longer does. The many queries that already filter it are
unchanged — cost if wrong: none.

**Ruling 5c — integer columns get integers.** `_write_trim_setup` wrote floats into the Integer
columns `points_ignored_start/end`. **Measured impact: zero** — SQLite stored all 45,143 as
integers, none non-integral. It now writes `int` when the value is integral, else NULL (the
raw value stays in `parameters`) — cost if wrong: none.

## 6. Closed by measurement, no code

- **Laser 3's per-position data is captured.** All 773 laser-3 (LTS3) passes carry cut length,
  trim current and used delta per position; in a 400-pass sample 99.8% of values are live and
  399 of 400 passes vary — the same as laser 2 (DLTS), whose sheets laser 3 writes.
- **`Response` vs `Response (Linear or Function)`** (laser 1 vs laser 2's label for what is
  probably one setting) stays split in storage; the setting sweep (TRACKER B1c) reads both
  under one name — why: a write-time alias would reach the 107,600 stored rows only through a
  reprocess — cost if wrong: none.

## Verification

- The whole suite: `python scripts/run_test_gate.py`, green.
- `tests/test_parse_all_models.py` (645 real files, every value): any entry that moves is
  refreshed alone, with the reason in the commit.
- Both QA sweeps on a COPY of the database; a new sweep entry for each new stored field.
- Every new filter, refusal and guard is made to fail first (mutation-checked).
- The back-fill script is run on a copy, and its result is checked against a fresh parse of the
  same files.
