# Unit-Level Yield

**Date:** 2026-05-08
**Status:** Design — awaiting user review

## Context

User exported model 8895 on 2026-05-08 to assess whether the model is trimming better over time. The export had 96 rows but only ~50 actual physical units, because:

- Each physical unit has a P (primary) and R (redundant) section, stored as two separate rows with serials like `3P` and `3R`
- Some units were retrimmed the same day, adding a third or fourth row to the same physical unit
- Junk serials (`TEST`, `x`, `test`) appear as "units" but aren't real

Quality rule from the user: **if any track on a unit fails, the unit fails and is scrapped.** So true yield is the fraction of physical units whose every section passed on its latest attempt.

The existing Trends "Linearity Pass Rate Trend" chart on the single-model detail view computes pass rate over rows (tracks), not units, so a model with a 50% section-level failure rate that always fails on the same unit (and is then retrimmed successfully) looks much worse than reality.

## Goal

Surface accurate unit-level yield in two places:

1. **In-app:** a "Unit Yield" trend line on the Trends single-model detail view, alongside the existing track-level P-chart.
2. **Excel:** a new "Yield by Unit" sheet on the per-model export.

Both share the same underlying definition of "unit" so they always agree.

## Out of scope

- **Final test (FT) records.** FT files encode the section in the filename (`Primary`/`Redundant`/`INNER`/`OUTER`) rather than the serial. The trim-table case is the v1 focus; FT is a v2 follow-on with its own filename-parsing logic.
- **Cost-per-unit, Cpk-per-unit, drift-per-unit.** Same rationale — start with the two surfaces the user actually needs today; promote to a column-everywhere approach only when more surfaces want it.
- **Dashboard pass-rate tile.** The Dashboard already uses a different denominator after the 2026-05-08 review fix. Switching it to unit-level later is a small change once the column exists.

## What gets stored

New nullable column `unit_id` on `analysis_results` (initially trim-table only).

Format: `"{model}/{shop_number}/{YYYY-MM-DD}"`. Examples from real 8895 data:

| serial | file_date | model | unit_id |
|---|---|---|---|
| `3P` | 2025-03-26 | 8895 | `8895/3/2025-03-26` |
| `3R` | 2025-03-26 | 8895 | `8895/3/2025-03-26` (same unit) |
| `3P` | 2025-03-26 17:43 | 8895 | `8895/3/2025-03-26` (retrim, same unit) |
| `4P` | 2025-03-26 | 8895 | `8895/4/2025-03-26` (different unit) |
| `TEST` | 2025-03-03 | 8895 | `NULL` (junk serial) |

## Shop-number extraction rule

Leading digits of `serial`. If serial is empty, `None`, or has no leading digit, `unit_id` is `NULL` and that row is excluded from yield calculations.

```python
def extract_shop_number(serial: str | None) -> str | None:
    if not serial:
        return None
    match = re.match(r'^\d+', serial.strip())
    return match.group(0) if match else None
```

Tested against these patterns from the 8895 data:

| serial | extracted shop | unit_id outcome |
|---|---|---|
| `3P` | `3` | grouped with `3R` |
| `1A` | `1` | grouped with `1` alone |
| `0009` | `0009` | leading-zero kept (string compare) |
| `8` | `8` | bare number |
| `196` | `196` | bare number |
| `TEST` | `None` | NULL, excluded |
| `x` | `None` | NULL, excluded |
| `` (empty) | `None` | NULL, excluded |

`file_date` formatted as `YYYY-MM-DD`. If `file_date` is NULL, `unit_id` is NULL.

## Unit pass rule

Applied at query time when grouping by `unit_id`. For each group:

- **Latest attempt per section.** If a section was retrimmed (e.g., `3P` appears twice on the same day), use the row with the highest `timestamp` as the section's authoritative status. `timestamp` is the row's write time, so it tracks the order operations actually occurred. `file_date` is the same for same-day retrims and would tie.
- **All-or-nothing.** A unit passes if every section's latest attempt is PASS. Any section status other than PASS (FAIL, WARNING, ERROR, UNTRIMMED) means the unit fails.

So `3P` (passed) + `3R` (failed) → unit fails. `3P` (failed) then `3P` (passed retrim) + `3R` (passed) → unit passes.

## Database migration

Two phases, both inside a transaction. Both idempotent.

**Phase 1: ALTER TABLE.** `ALTER TABLE analysis_results ADD COLUMN unit_id TEXT`. Wrapped in the existing migration helper that detects "duplicate column" and skips. Pre-flight log: "Adding unit_id column"; post-flight log: row count of `analysis_results`.

**Phase 2: Backfill.** A single UPDATE statement that computes `unit_id` from existing `model`, `serial`, `file_date`. Runs only for rows where `unit_id IS NULL` so it's safe to re-run. Done as a single SQL statement (SQLite handles 80k rows in seconds) using `CASE WHEN regexp...` — or, more portably, iterate in Python.

Python iteration is safer because the shop-number regex lives in Python. The migration script reads `(id, model, serial, file_date)` for rows with `unit_id IS NULL`, computes the unit_id in Python, and writes back in batches of 1000.

**Sanity checks logged after backfill:**

- `COUNT(*) WHERE unit_id IS NULL` — rows that had unparseable serials or missing date; expected nonzero (junk rows)
- `COUNT(*) WHERE unit_id IS NOT NULL` — rows that got a valid unit_id; should be ~95% of the table
- `COUNT(DISTINCT unit_id)` — distinct units
- Three sample unit_ids printed for spot-check

If any unexpected condition is hit (e.g., zero rows updated when many were expected, or backfill duration > 60s), the migration logs a warning but does not roll back — the column is nullable, the app keeps working.

## Write-time computation

In `_map_analysis_to_db` (`database/manager.py`), compute `unit_id` from `analysis.metadata.model`, `analysis.metadata.serial`, `analysis.metadata.file_date` before constructing the `DBAnalysisResult`. Pass it as `unit_id=unit_id` to the constructor.

Same logic added to `_update_existing_analysis` so re-processed files get their unit_id refreshed.

A single helper `compute_unit_id(model, serial, file_date) -> Optional[str]` lives near the shop-number extractor and is the single source of truth for both write-time and migration paths.

## In-app surface: Trends single-model detail

The existing Linearity Pass Rate P-chart stays. A second line overlay is added showing **Unit Yield over time**, computed as:

For each calendar day in the selected window:
1. Find all distinct `unit_id` values written that day for the selected model
2. For each unit, compute pass/fail via the unit pass rule above
3. Plot `passed_units / total_units * 100` as a percent

The two lines on the same chart let the operator see the divergence — track-level pass rate (sensitive to retrims) vs unit-level yield (sensitive to whether anything actually shipped).

The legend distinguishes them clearly. If unit-yield computation yields no qualifying units in the window (all NULL unit_ids), the line is hidden and a note appears in the chart.

## New Excel sheet: "Yield by Unit"

Added to the per-model export (the `_create_all_results_sheet` flow). One row per unit, columns:

| Column | Source |
|---|---|
| Shop Number | extracted shop number |
| Date | unit_id's date portion |
| Sections | `P, R` / `P` / `R` / etc. — comma-separated section letters from member rows |
| Attempts | total number of rows in the group (P retrim adds 1) |
| P Final Status | PASS / FAIL / (absent if no P) |
| R Final Status | same for R |
| Unit Status | PASS if all sections PASS, else FAIL |
| First Trim | earliest `file_date` in group |
| Last Trim | latest `file_date` in group |

Sorted by Date descending, then Shop Number. A summary row at the bottom: total units, passed units, unit yield %.

This sheet sits alongside the existing "All Results" sheet — operators who want row-level detail still have it.

## Edge cases and safety

**Junk serials (`TEST`, `x`, etc.):** `unit_id` is NULL. These rows appear in the existing "All Results" sheet (unchanged) but are excluded from the new "Yield by Unit" sheet and from the in-app unit-yield line.

**Missing `file_date`:** Same — NULL unit_id, excluded from yield.

**Unit spans two calendar days (rare retrim that crosses midnight):** Treated as two separate units. Acceptable for v1 — the user said shop numbers don't repeat the same day, so cross-day is genuinely a new unit by their own rule.

**Single-section units:** A unit with only `P` (no `R` ever recorded) is included; its status = the latest P status. If the user genuinely runs single-winding parts, this is correct. If `R` is just missing because of a parsing miss elsewhere, the unit reflects whatever was captured.

**Section letter not P/R/A/B:** Section letter is whatever non-digit suffix follows the shop number (e.g., `3RC` → section `RC`). Sections column in the Excel sheet shows whatever letters appear.

**Feature flag:** A config flag `enable_unit_yield_view` (default `True` once shipped). If something looks wrong in production, set to `False` to hide the new chart line and skip the new Excel sheet. Existing exports / charts unaffected by the flag.

## Testing

Pre-deploy tests cover:

1. `extract_shop_number` and `compute_unit_id` against the 8 serial patterns above
2. Migration runs idempotently — calling it twice does not error and produces identical state
3. Backfill on a synthetic 1000-row table produces correct `unit_id` distribution
4. Unit pass rule with: all-P-pass / one-P-fail / P-retrim-then-pass / P-only-no-R / R-only-no-P / no-pass-no-fail (all WARNING) cases
5. Junk serials (`TEST`, `x`, ``, `None`) produce NULL unit_id and are excluded from yield queries

## Success criteria

- After migration runs, ~95% of existing 80k trim rows have a non-NULL `unit_id`
- For model 8895: `SELECT COUNT(DISTINCT unit_id) FROM analysis_results WHERE model='8895' AND unit_id IS NOT NULL` returns roughly half the row count (~48 units from 96 rows)
- Trends single-model detail page shows two lines: one for row-level pass rate (unchanged) and one for unit-level yield
- Excel model export contains a new "Yield by Unit" sheet with one row per unit
- Feature flag toggling off the chart line and Excel sheet leaves the rest of the app unchanged
- All existing tests still pass; new tests cover the 8 serial patterns and the migration idempotency

## Architecture summary

| Component | Change |
|---|---|
| `database/models.py` | New `unit_id` column on `AnalysisResult` (nullable, indexed) |
| `database/manager.py` | New `compute_unit_id` helper; ALTER + backfill migration; write-time computation in `_map_analysis_to_db` and `_update_existing_analysis`; new query method `get_unit_yield_trend` for the chart |
| `gui/pages/trends.py` | Single-model detail view P-chart adds a second line for unit yield |
| `export/excel.py` | New `_create_yield_by_unit_sheet` called from the batch export |
| `config.py` | New flag `enable_unit_yield_view` |
| `tests/test_unit_yield.py` | New: extraction, computation, migration, pass rule, yield query |
