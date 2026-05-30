# Spec 1 — Upstream Signals Capture

**Date:** 2026-05-30
**Status:** Approved, ready for implementation plan
**Parent spec:** `docs/superpowers/specs/2026-05-14-app-mission-realignment-design.md` (anchor)
**Target branch:** `main` (Spec 1 is decision-independent of Q1; V6 inherits at next merge-forward)
**Estimated effort:** 1 short session — pure additive change, no behavior change to existing flows.

---

## Mission

Capture the per-track **untrimmed sigma gradient** as a new persisted metric so the multi-metric drift detector (Spec 2) can watch upstream element-production drift directly, without waiting for the post-trim signal to lag.

That's it. This spec stops at "the data is in the DB and gets populated on every new file processed." Drift detection on the new column, UI exposure, and tier-threshold tightening all live in downstream specs.

## Why this is Spec 1

Per the anchor doc (line 169) the current `sigma_gradient` is calculated only on post-trim `errors`. The untrimmed arrays are loaded into the analyzer's locals and persisted as JSON columns (`untrimmed_positions`, `untrimmed_errors` at `database/models.py:386-387`) but never themselves run through `_calculate_sigma()`. By the time post-trim sigma drifts, the upstream element-production process has been off for a while. The fix is one analyzer call + one column. Foundational for Spec 2.

Spec 1 is **decision-independent of Q1** because the column is useful to V5 and V6 alike. Landing on `main` means V6 picks it up free at the next merge-forward.

## Scope

### In scope
- New nullable `TrackResult.untrimmed_sigma_gradient` column.
- New `TrackData.untrimmed_sigma_gradient` Pydantic field.
- Analyzer change: parallel `_calculate_sigma` call on the untrimmed arrays inside `analyze_track`.
- DB save mapping (TrackData → TrackResult).
- Idempotent ALTER TABLE migration at startup matching the existing pattern (`UPGRADE_TRACKER.md` Task 1.3 added 24 indexes via `CREATE INDEX IF NOT EXISTS` at startup; we use the same idiom for the new column via `PRAGMA table_info` + conditional `ALTER TABLE`).
- Unit tests for the calculation, the DB persistence, and the migration on a pre-existing DB.

### Out of scope (explicitly deferred)
- **Drift detection on the new column** — Spec 2.
- **UI surfacing** — Spec 3 (per-model investigation view).
- **Backfill of historical rows.** Existing rows stay `NULL` until the file is reprocessed. James already routinely reprocesses; CLAUDE.md treats reprocess as the normal mechanism. No batch backfill job in this spec.
- **Pass/fail threshold for untrimmed sigma.** Untrimmed sigma is a *drift* signal, not a manufacturing spec. No `untrimmed_sigma_pass`, no `untrimmed_sigma_threshold`. Drift detection (Spec 2) compares the value to its own historical baseline.
- **Per-model ML threshold for untrimmed sigma.** Same reasoning. The existing per-model `_get_threshold` infrastructure is post-trim-specific.

## Technical approach

### 1. Schema — `src/laser_trim_analyzer/database/models.py`

Add one column to `TrackResult` (the class at line 310) immediately after the existing `sigma_gradient` definition at line 336:

```python
sigma_gradient = Column(Float, nullable=True)                        # existing
untrimmed_sigma_gradient = Column(Float, nullable=True)              # new
```

Add a parallel index near line 422 (next to `idx_track_sigma_gradient`):

```python
Index('idx_track_untrimmed_sigma_gradient', 'untrimmed_sigma_gradient'),
```

Add a parallel check constraint near line 432 (next to `check_sigma_gradient_positive`):

```python
CheckConstraint('untrimmed_sigma_gradient >= 0 OR untrimmed_sigma_gradient IS NULL',
                name='check_untrimmed_sigma_gradient_non_negative'),
```

(Note: existing constraint is `'sigma_gradient >= 0'` without the NULL clause — SQLite treats `NULL >= 0` as `NULL` which passes a check constraint by default, but the explicit `IS NULL` form is clearer about the intent here.)

Add a parallel validator near line 450:

```python
@validates('untrimmed_sigma_gradient')
def validate_untrimmed_sigma_gradient(self, key, value):
    """Validate untrimmed_sigma_gradient is non-negative when present."""
    if value is not None and value < 0:
        raise ValueError("untrimmed_sigma_gradient cannot be negative")
    return value
```

### 2. Pydantic model — `src/laser_trim_analyzer/core/models.py`

Add one field to `TrackData` (class at line 86) immediately after the existing `sigma_gradient` field at line 101:

```python
sigma_gradient: Optional[float] = Field(None, ge=0, description="Sigma gradient value")
untrimmed_sigma_gradient: Optional[float] = Field(
    None, ge=0,
    description="Sigma gradient calculated on untrimmed (pre-trim) arrays. "
                "Upstream element-quality signal; independent of post-trim "
                "process. NULL when untrimmed arrays absent or all-NaN.",
)
```

No corresponding `sigma_pass` / `sigma_threshold` companion fields (per the out-of-scope decision above).

### 3. Analyzer — `src/laser_trim_analyzer/core/analyzer.py`

In `analyze_track`, after the existing post-trim sigma call at line 213:

```python
sigma_gradient, sigma_threshold = self._calculate_sigma(
    positions, errors, linearity_spec, travel_length, unit_length, model
)
sigma_pass = sigma_gradient <= sigma_threshold
```

Add an untrimmed parallel call, gated on data availability and exception-safe:

```python
# Untrimmed (upstream) sigma — independent signal for Spec 2 drift
# detection.  Not used for pass/fail.  Gated on data availability;
# exception-safe because the per-track save must still proceed even if
# the untrimmed arrays are malformed.
untrimmed_sigma_gradient: Optional[float] = None
untrimmed_positions = track_data.get("untrimmed_positions") or []
untrimmed_errors_local = track_data.get("untrimmed_errors") or []
if untrimmed_positions and untrimmed_errors_local:
    try:
        # Filter NaN consistently with _calculate_trim_effectiveness:1050.
        valid_pairs = [
            (p, e)
            for p, e in zip(untrimmed_positions, untrimmed_errors_local)
            if p is not None and e is not None
            and not np.isnan(p) and not np.isnan(e)
        ]
        if len(valid_pairs) > 2 * END_POINT_FILTER_COUNT + 3:
            up = [p for p, _ in valid_pairs]
            ue = [e for _, e in valid_pairs]
            # Reuse the same threshold-context inputs; we only consume
            # the gradient half of the tuple.  Threshold for untrimmed
            # sigma is not a manufacturing spec — see spec out-of-scope.
            sig, _threshold = self._calculate_sigma(
                up, ue, linearity_spec, travel_length, unit_length, model
            )
            untrimmed_sigma_gradient = sig
    except Exception as e:
        logger.warning(
            f"Untrimmed sigma calculation failed for track "
            f"{track_id!r}: {e}; storing NULL"
        )
        untrimmed_sigma_gradient = None
```

Then pass through to `TrackData` at the existing return (line 284 onward), right next to `sigma_gradient=sigma_gradient`:

```python
sigma_gradient=sigma_gradient,
sigma_threshold=sigma_threshold,
sigma_pass=sigma_pass,
untrimmed_sigma_gradient=untrimmed_sigma_gradient,    # new
```

#### Edge cases (must be covered by tests)

| Condition | Stored value |
|---|---|
| Both `untrimmed_positions` and `untrimmed_errors` empty/absent | `NULL` |
| Arrays present but all values NaN | `NULL` (after filter, `valid_pairs` is empty) |
| Arrays present but fewer than `2*END_POINT_FILTER_COUNT + 3` valid points | `NULL` (matches the existing `_calculate_sigma` short-array path which returns `0.0`, but we want NULL not 0.0 to distinguish "not measured" from "measured as zero" — that's why the length gate is in the caller, not inside `_calculate_sigma`) |
| Arrays present and well-formed | computed sigma value |
| `_calculate_sigma` raises | `NULL`, with a `logger.warning` for triage |
| File is `UNTRIMMED`-only status (no post-trim curve) | Only `untrimmed_sigma_gradient` populated; `sigma_gradient` stays `NULL` — the existing UNTRIMMED-only code path remains responsible for that, this spec doesn't change it |

### 4. DB save mapping

The mapping from `TrackData` → `TrackResult` is currently in `database/manager.py` (the trim save path). Field assignment is field-by-field. Locate the `db_track = TrackResult(...)` (or equivalent setattr loop) for trim tracks and add the line:

```python
untrimmed_sigma_gradient=track.untrimmed_sigma_gradient,
```

If the codebase uses a `**asdict(track)` style splat, no change is needed — the new field flows automatically.

**Implementer task:** grep for `sigma_gradient=` in `database/manager.py`, find the trim-track build site(s), and mirror the assignment.

### 5. Startup migration

Existing convention (per `docs/UPGRADE_TRACKER.md` Task 1.3): idempotent migrations run at app startup via `CREATE INDEX IF NOT EXISTS` style statements. For ADD COLUMN, SQLite requires a `PRAGMA table_info` probe + conditional ALTER TABLE because `IF NOT EXISTS` isn't supported on `ALTER TABLE ADD COLUMN`. The codebase already has this pattern — `database/manager.py` lines 758, 772, 787, 797, 806, 816 (per the Task 6 grep results) all show probe-then-add-column patterns wrapped in `if "duplicate column" not in str(e).lower() and "already exists" not in str(e).lower(): ...` guards.

Add a single new migration entry following the same idiom:

```python
# Add untrimmed_sigma_gradient to track_results (Spec 1 / 2026-05-30).
try:
    session.execute(text(
        "ALTER TABLE track_results ADD COLUMN untrimmed_sigma_gradient FLOAT"
    ))
    session.commit()
except Exception as e:
    session.rollback()
    if "duplicate column" not in str(e).lower() and "already exists" not in str(e).lower():
        raise
    # else: column already exists from a prior startup — fine.
```

Also add the corresponding `CREATE INDEX IF NOT EXISTS idx_track_untrimmed_sigma_gradient ON track_results (untrimmed_sigma_gradient)` migration in the same block where the other indexes are created (so the index exists for drift queries even on databases that pre-date Spec 1).

### 6. Tests

Add a new test module `tests/test_spec1_untrimmed_sigma.py`:

1. **Calculation populated when arrays present.** Build a `TrackData`-shaped input with realistic untrimmed arrays (≥ 20 points, non-NaN), run through `analyze_track`, assert `result.untrimmed_sigma_gradient` is a finite positive float.
2. **NULL when untrimmed arrays absent.** Same call but with `untrimmed_positions=None`/`untrimmed_errors=None`. Assert `result.untrimmed_sigma_gradient is None`.
3. **NULL when arrays all NaN.** Assert NULL.
4. **NULL when arrays too short** (< `2*END_POINT_FILTER_COUNT + 3` valid points after NaN filter). Assert NULL.
5. **Persistence round-trip.** Save through `DatabaseManager`, re-query the row, assert `untrimmed_sigma_gradient` matches.
6. **Migration on pre-existing DB.** Create a fresh DB *without* the new column (use a fixture that builds the schema manually missing that column), then construct `DatabaseManager` against it. Assert the startup migration adds the column without error, and a subsequent save populates it.
7. **Existing `sigma_gradient` unchanged.** Assert that before/after Spec 1, the post-trim `sigma_gradient` value on the same input is bit-identical (no regression in the existing calculation).
8. **`UNTRIMMED`-only status path still produces `sigma_gradient = NULL`.** Just confirm the existing behavior; the test from `tests/test_drift_dashboard_data.py:347-354` already covers this but a parallel assertion here keeps Spec 1's contract self-contained.

## Success criteria

1. After this spec lands, every new file processed via the existing `process_file` / batch flow produces a `TrackResult` row whose `untrimmed_sigma_gradient` is either a positive float or `NULL`. Never a negative, never an error.
2. Existing `sigma_gradient`, `sigma_threshold`, `sigma_pass` values are unchanged for every input.
3. A SQL query `SELECT model, COUNT(*) FROM track_results WHERE untrimmed_sigma_gradient IS NOT NULL GROUP BY model` returns useful data within a session of reprocessing one batch (verifies the column populates).
4. Restarting the app against a database created before Spec 1 succeeds (migration runs idempotently).

## Downstream impact (informational — what Spec 2 needs to know)

- The new column is **nullable** and many historical rows will be `NULL` until reprocessed. Spec 2's drift detector must treat `NULL` as "no observation," not zero.
- `UNTRIMMED`-only records (no laser pass at all) carry `untrimmed_sigma_gradient` but not `sigma_gradient`. Spec 2's per-metric state must accept records that contribute to one metric's series but not another.
- Per-model ML threshold infrastructure is **not** extended to the new column in this spec. Spec 2's per-metric CUSUM/EWMA derives its own baseline from the historical untrimmed sigma values; no threshold-from-formula or threshold-from-ML wiring is needed.

## Non-goals

- Do not change `_calculate_sigma`'s signature or behavior.
- Do not change the post-trim sigma calculation site (line 213) — only add a parallel call below it.
- Do not add UI surfaces for the new column. Spec 3 owns that.
- Do not add a CLI / one-shot script to backfill historical rows. Reprocessing already does this naturally.
- Do not export `untrimmed_sigma_gradient` in Excel exports until requested. (If James asks, that's a one-line add in `export/excel.py` and can be a follow-up.)
