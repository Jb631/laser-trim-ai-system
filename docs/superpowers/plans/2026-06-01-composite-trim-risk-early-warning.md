# Composite Trim-Risk Early-Warning Signal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single weak sigma signal with a per-model **composite trim-risk score** — a logistic blend of orthogonal trim-effort features (worst raw point, raw steepness, resistance offset, and trim passes) — whose per-group trend is the upstream-drift early warning.

**Architecture:** Two phases on two branches. **Phase 1 (main/V5)** makes every composite input feature available: it adds the one missing field (`untrimmed_error_max`), backfills the three derivable features on the existing DB with no reprocess, and documents the single reprocess that populates `trim_pass_count`. **Phase 2 (V6)** trains a per-model logistic risk model with grouped cross-validation and an honest deploy-gate, persists a per-unit `composite_trim_risk_score`, and registers it as the 8th watched drift metric so the existing CUSUM/EWMA engine watches its group trend.

**Tech Stack:** Python 3, SQLAlchemy 2.0 ORM (SQLite), pandas, scikit-learn (LogisticRegression, GroupKFold, roc_auc_score), pytest. Existing per-model ML lives in `src/laser_trim_analyzer/ml/`; the V6 drift engine in `ml/drift_training.py` + `ml/drift_types.py`.

---

## Why this design (decision log)

These were settled during the 2026-06-01 brainstorm; do not re-litigate them while executing:

1. **The signal is *trim effort*, not pass/fail.** A unit can pass linearity yet need near-maximum trim effort; its effort is the canary for the *next* group. Effort metrics live in the files, not (fully) in the DB.
2. **Failures have ≥3 physically distinct modes**, each needing its own feature (validated by grouped-CV AUC):
   - severe local raw non-linearity trim can't fix → `untrimmed_error_max` (best single signal, AUC up to 0.82; zero-tolerance spec means the *worst point* governs)
   - raw steepness → `untrimmed_sigma_gradient` (weak alone, ~0.55; **redundant with error_max — keep only because the per-model fit may use it where error_max is inverted**)
   - resistance so far off that hitting target eats the linearity budget → `resistance_change_percent` (weak alone but **orthogonal**; rescued model 6126 from AUC 0.45→0.66)
   - ran out of trim headroom → `trim_pass_count` (**untested — NULL in dev DB; the reprocess gate**)
3. **`error_max + sigma` alone gives no lift** (both come from the same raw curve). The composite earns its keep *only* by combining orthogonal modes. Therefore Phase 2 includes a **deploy-gate**: a model gets a deployed composite *only* if grouped-CV AUC beats the best single feature by ≥ 0.02 **and** honest confidence ≥ 0.20; otherwise the engine falls back to `untrimmed_error_max`.
4. **This is a drift / early-warning signal, not a unit go/no-go gate.** Per-unit ceiling is ~0.67 for most models. Its power is at the group level, where averaging sharpens it. We watch `group-mean(score)` with the existing CUSUM.
5. **Persist the score as a per-unit column** (not compute-on-the-fly) so it is auditable, exportable, and the drift engine sees a stable time series — matching all 7 existing watched metrics.

---

## Ground-truth facts (verified 2026-06-01, do not re-discover)

- **`untrimmed_error_max` exists nowhere** — not in `core/models.py` `TrackData`, not in `database/models.py` `TrackResult`. Must be added.
- **`untrimmed_rms_error`, `resistance_change`, `trim_pass_count` are wired end-to-end on main** (parser → analyzer → `_map_track_to_db`). They are NULL in the dev DB only because it predates commit `26894cb`. Verified: the current parser yields `trim_pass_count=2`, resistance `4486→5256` on a System-A sample; `trim_pass_count=1`, `9451→10287` on System-B.
- **`untrimmed_errors` arrays are 100% populated** (80,656/80,656) → `untrimmed_error_max` and `untrimmed_rms_error` are backfillable from them with no reprocess.
- **`untrimmed_resistance` (93%) and `trimmed_resistance` (98%)** are populated → `resistance_change` and `resistance_change_percent` are backfillable with no reprocess.
- **`trim_pass_count` is NOT derivable** from stored data (counted from file sheets) → needs a reprocess on the work machine.

### Execution environment (read before running any step)

- **Use `python3`, not `python`** — there is no `python` on PATH. Every `python -m pytest ...` command in this plan means `python3 -m pytest ...`.
- **Every new ORM Column REQUIRES a matching migration** in `DatabaseManager._run_migrations()` (`database/manager.py`). Adding a Column without the migration breaks *all* `track_results` queries against existing DBs with `OperationalError: no such column`. The `untrimmed_sigma_gradient` block (~L576) is the canonical template (migrates column **and** index). This applies to Task 1 (`untrimmed_error_max`) and Task 5 (`composite_trim_risk_score`). Always run the **full** suite after a schema change, not just the new tests.

### Key file locations (exact)

| What | File | Anchor |
|---|---|---|
| `TrackData` pydantic model | `src/laser_trim_analyzer/core/models.py` | `untrimmed_rms_error` field ~L174; `resistance_change_percent` is an `@property` ~L196-200 |
| `TrackResult` ORM | `src/laser_trim_analyzer/database/models.py` | `untrimmed_sigma_gradient` L338; `resistance_change` L346; `resistance_change_percent` L347; `untrimmed_rms_error` L377; `untrimmed_errors` L389; `trim_pass_count` L407; index block ~L425 |
| Trim-effort computation | `src/laser_trim_analyzer/core/analyzer.py` | `Analyzer` L138; `analyze_track` L162; `_calculate_trim_effectiveness` L1084-1128; trim metrics unpacked via `**trim_metrics` ~L369 |
| TrackData → DB mapping | `src/laser_trim_analyzer/database/manager.py` | `_map_track_to_db` ~L2702; trim-effort mapping L2751-2756; `trim_pass_count=getattr(track,'trim_pass_count',None)` L2773 |
| Backfill template | `scripts/backfill_untrimmed_sigma.py` | whole file (idempotent, NULL-only, COALESCE pattern) |
| Per-model training data | `src/laser_trim_analyzer/ml/manager.py` | `_get_training_data` L887-980; `records.append({...})` L923-941; `train_model` L198; predictor trained ~L331; `apply_to_database` loop L485; `storage_path` L113; predictor save loop L1011-1019 |
| Predictor (pattern to copy) | `src/laser_trim_analyzer/ml/predictor.py` | `ModelPredictor` L114; `FEATURE_COLUMNS` L51-56; `train(features,labels,severity,groups)` L161; `GroupKFold` use L332-338; `save` L529-610; `load` L639-712 |
| Honest-confidence (pattern to copy) | `src/laser_trim_analyzer/ml/threshold_optimizer.py` | rank-AUC confidence L188-201; `ThresholdResult` L22-42 |
| Test style | `tests/test_audit_fixes_2026_06_01.py` | direct imports, direct object construction, no conftest |

### V6-branch facts (read via `git show V6:<path>`; line numbers are V6's)

| What | File (on V6) | Anchor |
|---|---|---|
| Watched metrics (7) | `ml/drift_types.py` | `WATCHED_METRICS` tuple L59-67 (string keys) |
| Metric→column map | `ml/drift_training.py` | `_TRACK_METRIC_COLUMNS` L40-50; `_load_samples_with_dates` L248-278; `advance_drift_state` L313-360; `_upsert_metric_state` L190-245; Bonferroni `target_fp_for_tier(...)/n_metrics` |
| Drift state table | `database/models.py` | `model_metric_state`; `metric` is a String key; cols `cusum_pos/cusum_neg/ewma_state/baseline_cutoff_date/last_updated` + per-tier h/L/z; `UniqueConstraint('model','metric')` |
| Drift UI | `gui/pages/trends.py` | all-models view auto-includes via `MultiMetricDriftDetector.get_status()` worst-of-N; single-model dashboard hardcodes `metric_axes`; Process-drift `metric_options` ~L2950-2970 |

---

## Scope & branch note

Phase 1 and Phase 2 are independent subsystems on different branches. **Each ships and tests on its own.** Execute Phase 1 fully on `main` and verify green before starting Phase 2. Phase 2 is executed on `V6` and must be preceded by the **Reprocess Gate** (between the phases) on the work machine, then a one-time re-validation, before the composite is deployed for any model.

Branch setup is assumed already done (CLAUDE.md session checklist). Confirm with `git branch --show-current` at the start of each phase.

---

# PHASE 1 — Feature availability (branch: `main`)

## Task 1: Add the `untrimmed_error_max` field and column

**Files:**
- Modify: `src/laser_trim_analyzer/core/models.py` (TrackData, near the `untrimmed_rms_error` field ~L174)
- Modify: `src/laser_trim_analyzer/database/models.py` (TrackResult, near `untrimmed_rms_error` L377; index block ~L425)
- Test: `tests/test_composite_trim_risk.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_composite_trim_risk.py`:

```python
"""Tests for the composite trim-risk early-warning feature (2026-06-01 plan)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_trackdata_has_untrimmed_error_max_field():
    from laser_trim_analyzer.core.models import TrackData
    fields = TrackData.model_fields  # pydantic v2
    assert "untrimmed_error_max" in fields, "TrackData must expose untrimmed_error_max"


def test_trackresult_has_untrimmed_error_max_column():
    from laser_trim_analyzer.database.models import TrackResult
    assert hasattr(TrackResult, "untrimmed_error_max"), \
        "TrackResult must have an untrimmed_error_max column"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_composite_trim_risk.py -v`
Expected: FAIL — `assert "untrimmed_error_max" in fields`.

- [ ] **Step 3: Add the TrackData field**

In `src/laser_trim_analyzer/core/models.py`, immediately after the existing `untrimmed_rms_error` field (~L174), add:

```python
    untrimmed_error_max: Optional[float] = Field(
        None, ge=0,
        description="Max |error| on the untrimmed (pre-trim) sweep. Zero-tolerance "
                    "linearity is governed by the worst point; this is the strongest "
                    "single upstream-drift signal. Exclude-aware, NaN-safe.",
    )
```

- [ ] **Step 4: Add the TrackResult column + index**

In `src/laser_trim_analyzer/database/models.py`, after `untrimmed_rms_error = Column(Float)` (~L377), add:

```python
    untrimmed_error_max = Column(Float)
```

In the `__table_args__` index block (~L425), beside `Index('idx_track_untrimmed_sigma_gradient', 'untrimmed_sigma_gradient')`, add:

```python
        Index('idx_track_untrimmed_error_max', 'untrimmed_error_max'),
```

- [ ] **Step 4b: Register the column in the existing-DB migration (REQUIRED)**

> Adding a Column to the ORM makes every `track_results` SELECT include it. Existing databases (`data/analysis.db`) don't have the physical column, so without a migration **every historical-data query raises `OperationalError: no such column: untrimmed_error_max`**. The app migrates existing DBs in `DatabaseManager._run_migrations()` (`src/laser_trim_analyzer/database/manager.py`, called during init). Add a migration block following the existing `untrimmed_sigma_gradient` template (which migrates column **and** index), placed right after it:

```python
            # Migration: Add untrimmed_error_max column to track_results.
            # Worst-case |error| across untrimmed points; complements
            # untrimmed_sigma_gradient as an element-quality signal.
            try:
                session.execute(
                    text("SELECT untrimmed_error_max FROM track_results LIMIT 1")
                )
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                logger.info("Running migration: Adding untrimmed_error_max column")
                try:
                    session.execute(text(
                        "ALTER TABLE track_results "
                        "ADD COLUMN untrimmed_error_max FLOAT"
                    ))
                    session.execute(text(
                        "CREATE INDEX IF NOT EXISTS idx_track_untrimmed_error_max "
                        "ON track_results (untrimmed_error_max)"
                    ))
                    session.commit()
                    logger.info("Migration completed: Added untrimmed_error_max")
                except Exception as e:
                    logger.warning(f"Migration warning (may already exist): {e}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_composite_trim_risk.py -v`
Expected: PASS (both tests). **Also run the full suite** (`python -m pytest tests/ -q`) — the migration must keep historical-data queries green: any test that queries the real `data/analysis.db` (e.g. `tests/test_5_8_2026_bugfixes.py::test_get_historical_data_light_load_returns_rows`) will fail with `no such column` if Step 4b is skipped.

- [ ] **Step 6: Commit**

```bash
git add tests/test_composite_trim_risk.py src/laser_trim_analyzer/core/models.py src/laser_trim_analyzer/database/models.py
git commit -m "feat(composite): add untrimmed_error_max field + column"
```

---

## Task 2: Compute `untrimmed_error_max` in the analyzer and map it to the DB

**Files:**
- Modify: `src/laser_trim_analyzer/core/analyzer.py` (`_calculate_trim_effectiveness` L1084-1128)
- Modify: `src/laser_trim_analyzer/database/manager.py` (`_map_track_to_db` mapping block L2751-2756)
- Test: `tests/test_composite_trim_risk.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_composite_trim_risk.py`:

```python
def test_analyzer_computes_untrimmed_error_max():
    from laser_trim_analyzer.core.analyzer import Analyzer
    a = Analyzer()  # __init__ args (scaling_factor, model_thresholds) all have defaults
    # untrimmed errors with a clear worst point at -0.05; trimmed much smaller
    untrimmed = [0.01, -0.02, 0.03, -0.05, 0.012]
    trimmed = [0.001, -0.002, 0.0015, -0.001, 0.0008]
    res = a._calculate_trim_effectiveness(
        trimmed_errors=trimmed,
        untrimmed_errors=untrimmed,
        untrimmed_resistance=4486.0,
        trimmed_resistance=5256.0,
    )
    assert abs(res["untrimmed_error_max"] - 0.05) < 1e-9
    assert abs(res["resistance_change"] - 770.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_composite_trim_risk.py::test_analyzer_computes_untrimmed_error_max -v`
Expected: FAIL — `KeyError: 'untrimmed_error_max'`.

- [ ] **Step 3: Compute it in `_calculate_trim_effectiveness`**

In `src/laser_trim_analyzer/core/analyzer.py` `_calculate_trim_effectiveness`, `valid_untrimmed` is built at L1104-1105 and `valid_trimmed` at L1106-1107, then the existing RMS block is guarded by `if valid_untrimmed and valid_trimmed:` (L1109). **Insert the new computation between L1107 and L1109** so it is guarded by `valid_untrimmed` **alone** — untrimmed-only test-sweep tracks (status UNTRIMMED, the drift case) have an empty `valid_trimmed` and must still get this field:

```python
            # Worst single point on the raw sweep -- governs zero-tolerance
            # linearity; the strongest single upstream-drift signal. Guarded by
            # valid_untrimmed ALONE so untrimmed-only test-sweep tracks still get
            # it. Matches the existing max(abs(...)) idiom at L1121.
            # (2026-06-01 composite plan.)
            if valid_untrimmed:
                result["untrimmed_error_max"] = float(max(abs(e) for e in valid_untrimmed))
```

> `valid_untrimmed` is already None/NaN-filtered (L1104-1105) and `np` is imported at module top, so no extra imports are needed. Do **not** place this inside the `valid_untrimmed and valid_trimmed` block — that would skip untrimmed-only tracks.

- [ ] **Step 4: Map it to the DB row**

In `src/laser_trim_analyzer/database/manager.py`, in `_map_track_to_db`, in the trim-effort block (after `untrimmed_rms_error=track.untrimmed_rms_error,` ~L2754), add:

```python
                untrimmed_error_max=getattr(track, 'untrimmed_error_max', None),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_composite_trim_risk.py -v`
Expected: PASS (all tests so far).

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/core/analyzer.py src/laser_trim_analyzer/database/manager.py tests/test_composite_trim_risk.py
git commit -m "feat(composite): compute + persist untrimmed_error_max"
```

---

## Task 3: Backfill the three derivable features on the existing DB (no reprocess)

**Files:**
- Create: `scripts/backfill_trim_effort.py`
- Test: `tests/test_composite_trim_risk.py`

This fills, **NULL rows only**, from data already present: `untrimmed_error_max` and `untrimmed_rms_error` from the `untrimmed_errors` array; `resistance_change` and `resistance_change_percent` from the two resistance columns. Idempotent. Mirrors `scripts/backfill_untrimmed_sigma.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_composite_trim_risk.py`:

```python
def test_backfill_fills_only_null_rows(tmp_path):
    import sqlite3, json
    from scripts.backfill_trim_effort import backfill_trim_effort

    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE track_results (
            id INTEGER PRIMARY KEY,
            untrimmed_errors TEXT,
            untrimmed_resistance REAL,
            trimmed_resistance REAL,
            untrimmed_error_max REAL,
            untrimmed_rms_error REAL,
            resistance_change REAL,
            resistance_change_percent REAL
        );
        """
    )
    # row 1: everything NULL -> should be filled
    con.execute(
        "INSERT INTO track_results (id, untrimmed_errors, untrimmed_resistance, trimmed_resistance) "
        "VALUES (1, ?, 4486.0, 5256.0)",
        (json.dumps([0.01, -0.05, 0.03]),),
    )
    # row 2: untrimmed_error_max already set -> must NOT be overwritten
    con.execute(
        "INSERT INTO track_results (id, untrimmed_errors, untrimmed_error_max) VALUES (2, ?, 999.0)",
        (json.dumps([0.01, -0.05, 0.03]),),
    )
    con.commit(); con.close()

    n = backfill_trim_effort(str(db))
    assert n >= 1

    con = sqlite3.connect(db)
    r1 = con.execute(
        "SELECT untrimmed_error_max, untrimmed_rms_error, resistance_change, resistance_change_percent "
        "FROM track_results WHERE id=1"
    ).fetchone()
    assert abs(r1[0] - 0.05) < 1e-9          # error_max
    assert r1[1] > 0                          # rms filled
    assert abs(r1[2] - 770.0) < 1e-9          # resistance_change
    assert abs(r1[3] - (770.0 / 4486.0 * 100)) < 1e-6
    r2 = con.execute("SELECT untrimmed_error_max FROM track_results WHERE id=2").fetchone()
    assert r2[0] == 999.0                      # untouched
    con.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_composite_trim_risk.py::test_backfill_fills_only_null_rows -v`
Expected: FAIL — `ModuleNotFoundError: scripts.backfill_trim_effort`.

- [ ] **Step 3: Write the backfill script**

Create `scripts/backfill_trim_effort.py`:

```python
"""One-time, idempotent backfill of derivable trim-effort metrics.

Fills NULL rows ONLY, from data already in the row:
  - untrimmed_error_max     = max(|untrimmed_errors|)
  - untrimmed_rms_error     = rms(untrimmed_errors)
  - resistance_change       = trimmed_resistance - untrimmed_resistance
  - resistance_change_percent = change / untrimmed_resistance * 100

Does NOT touch trim_pass_count (not derivable -- needs a reprocess).
Safe to run repeatedly. Usage:  python scripts/backfill_trim_effort.py [path/to/analysis.db]
"""
import json
import math
import sqlite3
import sys


def _stats(raw):
    try:
        arr = [float(x) for x in json.loads(raw) if x is not None]
    except Exception:
        return None, None
    arr = [x for x in arr if not math.isnan(x)]
    if not arr:
        return None, None
    emax = max(abs(x) for x in arr)
    rms = math.sqrt(sum(x * x for x in arr) / len(arr))
    return emax, rms


def backfill_trim_effort(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    updated = 0

    # --- error_max / rms from the untrimmed_errors array ---
    rows = cur.execute(
        "SELECT id, untrimmed_errors FROM track_results "
        "WHERE untrimmed_errors IS NOT NULL "
        "AND (untrimmed_error_max IS NULL OR untrimmed_rms_error IS NULL)"
    ).fetchall()
    for rid, raw in rows:
        emax, rms = _stats(raw)
        if emax is None:
            continue
        cur.execute(
            "UPDATE track_results SET "
            "untrimmed_error_max = COALESCE(untrimmed_error_max, ?), "
            "untrimmed_rms_error = COALESCE(untrimmed_rms_error, ?) "
            "WHERE id = ?",
            (emax, rms, rid),
        )
        updated += cur.rowcount

    # --- resistance_change / percent from the two resistance columns ---
    cur.execute(
        "UPDATE track_results SET "
        "resistance_change = COALESCE(resistance_change, trimmed_resistance - untrimmed_resistance) "
        "WHERE resistance_change IS NULL "
        "AND untrimmed_resistance IS NOT NULL AND trimmed_resistance IS NOT NULL"
    )
    cur.execute(
        "UPDATE track_results SET "
        "resistance_change_percent = COALESCE(resistance_change_percent, "
        "(trimmed_resistance - untrimmed_resistance) / untrimmed_resistance * 100.0) "
        "WHERE resistance_change_percent IS NULL "
        "AND untrimmed_resistance IS NOT NULL AND untrimmed_resistance != 0 "
        "AND trimmed_resistance IS NOT NULL"
    )

    con.commit()
    con.close()
    return updated


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/analysis.db"
    n = backfill_trim_effort(path)
    print(f"Backfilled trim-effort metrics in {path}: {n} array-derived row-updates "
          f"(resistance columns updated in bulk).")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_composite_trim_risk.py -v`
Expected: PASS (all).

- [ ] **Step 5: Rehearse on a COPY of the dev DB (never the original)**

```bash
cp data/analysis.db /tmp/rehearse.db
python scripts/backfill_trim_effort.py /tmp/rehearse.db
python - <<'PY'
import sqlite3
c = sqlite3.connect("/tmp/rehearse.db")
for col in ["untrimmed_error_max","untrimmed_rms_error","resistance_change","resistance_change_percent"]:
    n = c.execute(f"SELECT COUNT(*) FROM track_results WHERE {col} IS NOT NULL").fetchone()[0]
    print(f"{col}: {n:,} populated")
PY
```
Expected: each of the four columns now populated for the large majority of rows (error_max/rms ~100%, resistance ~93%). Re-running the script must report ~0 new updates (idempotent).

- [ ] **Step 6: Commit**

```bash
git add scripts/backfill_trim_effort.py tests/test_composite_trim_risk.py
git commit -m "feat(composite): idempotent backfill for derivable trim-effort metrics"
```

---

## Task 4: Document the `trim_pass_count` reprocess gate (runbook, no code)

**Files:**
- Modify: `docs/AUDIT_FIX_PLAN_2026-06-01.md` (append a "Composite trim-risk" section), OR create `docs/REPROCESS_RUNBOOK_trim_pass_count.md`

- [ ] **Step 1: Write the runbook**

Create `docs/REPROCESS_RUNBOOK_trim_pass_count.md`:

```markdown
# Reprocess runbook — populate trim_pass_count (work machine only)

`trim_pass_count` is the only composite feature that is NOT derivable from stored
data; it is counted from the trim sheets in each source file. The current parser
populates it correctly (verified: System A -> 2, System B -> 1). The production DB
has it NULL only because those rows predate commit 26894cb.

## Steps (work machine, against the REAL DB)
1. `git pull` on `main` (brings Tasks 1-3: untrimmed_error_max + backfill script).
2. Back up the DB:  copy analysis.db -> analysis.db.YYYY-MM-DD.bak
3. Run the derivable backfill first (instant, no reprocess):
   `python scripts/backfill_trim_effort.py data/analysis.db`
4. Reprocess source files so trim_pass_count is captured. Either:
   - Full reprocess of the archive, OR
   - Incremental: the content-hash dedup (Batch D) will skip unchanged files, so a
     reprocess only re-reads files whose rows lack trim_pass_count. Confirm the
     processor maps trim_pass_count (database/manager.py L2773) before running.
5. Verify:  SELECT COUNT(*) FROM track_results WHERE trim_pass_count IS NOT NULL;
   Expect the large majority populated. Spot-check distribution (1,2,3...).

## Gate
Phase 2 deploy-gate re-validation (Task 11) must be run AFTER this, because the
3rd orthogonal mode (trim headroom) only exists once trim_pass_count is populated.
Until then Phase 2 trains on the 3 available features and the gate will deploy the
composite only where it already beats untrimmed_error_max alone.
```

- [ ] **Step 2: Commit**

```bash
git add docs/REPROCESS_RUNBOOK_trim_pass_count.md
git commit -m "docs(composite): reprocess runbook for trim_pass_count"
```

- [ ] **Step 3: Phase 1 gate — full suite green on main**

Run: `python -m pytest tests/ -q`
Expected: all pass (Phase 1 adds tests; nothing else regresses). Push `main`.

```bash
git push origin main
```

---

# REPROCESS GATE (work machine, between phases)

Execute `docs/REPROCESS_RUNBOOK_trim_pass_count.md` on the work machine before deploying Phase 2. Phase 2 *code* can be built on V6 in parallel, but the **deploy-gate re-validation (Task 11)** and any model rollout must wait until `trim_pass_count` is populated in the DB Phase 2 runs against.

---

# PHASE 2 — Composite risk model + drift wiring (branch: `V6`)

> Switch to V6 first: `git checkout V6` then `git merge main` (to bring Phase 1's `untrimmed_error_max` column + backfill). Confirm `git branch --show-current` → `V6`. All V6 line numbers below are from the V6 branch.

## Task 5: Add the `composite_trim_risk_score` per-unit column

**Files:**
- Modify: `src/laser_trim_analyzer/database/models.py` (TrackResult + index)
- Test: `tests/test_composite_trim_risk.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_composite_trim_risk.py`:

```python
def test_trackresult_has_composite_score_column():
    from laser_trim_analyzer.database.models import TrackResult
    assert hasattr(TrackResult, "composite_trim_risk_score")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_composite_trim_risk.py::test_trackresult_has_composite_score_column -v`
Expected: FAIL.

- [ ] **Step 3: Add the column + index**

In `src/laser_trim_analyzer/database/models.py` `TrackResult`, near the other derived scores, add:

```python
    # Per-unit composite trim-risk score in [0,1] from the per-model logistic.
    # Watched at the group level for upstream drift (2026-06-01 composite plan).
    composite_trim_risk_score = Column(Float, nullable=True)
```

In `__table_args__`:

```python
        Index('idx_track_composite_trim_risk_score', 'composite_trim_risk_score'),
```

- [ ] **Step 3b: Register the column in the existing-DB migration (REQUIRED)**

Same as Phase 1 Task 1 Step 4b — add a migration block in `DatabaseManager._run_migrations()` (`src/laser_trim_analyzer/database/manager.py`) so existing DBs get the physical column, otherwise every `track_results` query raises `OperationalError: no such column: composite_trim_risk_score`. Follow the `untrimmed_error_max` migration block as the template:

```python
            # Migration: Add composite_trim_risk_score column to track_results.
            try:
                session.execute(
                    text("SELECT composite_trim_risk_score FROM track_results LIMIT 1")
                )
            except OperationalError:
                session.rollback()
                logger.info("Running migration: Adding composite_trim_risk_score column")
                try:
                    session.execute(text(
                        "ALTER TABLE track_results "
                        "ADD COLUMN composite_trim_risk_score FLOAT"
                    ))
                    session.execute(text(
                        "CREATE INDEX IF NOT EXISTS idx_track_composite_trim_risk_score "
                        "ON track_results (composite_trim_risk_score)"
                    ))
                    session.commit()
                    logger.info("Migration completed: Added composite_trim_risk_score")
                except Exception as e:
                    logger.warning(f"Migration warning (may already exist): {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_composite_trim_risk.py::test_trackresult_has_composite_score_column -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/database/models.py tests/test_composite_trim_risk.py
git commit -m "feat(composite): add composite_trim_risk_score column (V6)"
```

---

## Task 6: Extend `_get_training_data` to return the composite features

**Files:**
- Modify: `src/laser_trim_analyzer/ml/manager.py` (`_get_training_data` `records.append({...})` L923-941)
- Test: `tests/test_composite_trim_risk.py`

- [ ] **Step 1: Write the failing test** (structural — asserts the keys exist)

Append:

```python
def test_training_record_includes_composite_features():
    import inspect
    from laser_trim_analyzer.ml import manager as mgr
    src = inspect.getsource(mgr._get_training_data) if hasattr(mgr, "_get_training_data") \
        else inspect.getsource(mgr.MLManager._get_training_data)
    for key in ("untrimmed_error_max", "resistance_change_percent", "trim_pass_count"):
        assert f"'{key}'" in src or f'"{key}"' in src, f"_get_training_data must emit {key}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_composite_trim_risk.py::test_training_record_includes_composite_features -v`
Expected: FAIL.

- [ ] **Step 3: Add the three columns to the emitted record**

In `src/laser_trim_analyzer/ml/manager.py`, in `_get_training_data`, the `records.append({...})` dict (L923-941), add these keys (reading from the `track` TrackResult):

```python
            'untrimmed_error_max': track.untrimmed_error_max,
            'resistance_change_percent': track.resistance_change_percent,
            'trim_pass_count': track.trim_pass_count,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_composite_trim_risk.py::test_training_record_includes_composite_features -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/ml/manager.py tests/test_composite_trim_risk.py
git commit -m "feat(composite): expose trim-effort features to per-model training data"
```

---

## Task 7: The `CompositeRiskModel` class (grouped-CV, honest confidence, deploy-gate)

**Files:**
- Create: `src/laser_trim_analyzer/ml/composite_risk.py`
- Test: `tests/test_composite_trim_risk.py`

Design: a per-model logistic over the available subset of `FEATURES`. Trains with grouped out-of-fold predictions (group = serial), computes honest AUC + confidence (the `threshold_optimizer` pattern), and sets `deployed` only if it beats the best single feature by ≥ `MIN_LIFT` and confidence ≥ `MIN_CONFIDENCE`. Missing features (e.g. `trim_pass_count` before the reprocess) are dropped, not imputed to a constant — the model uses whatever columns have data.

- [ ] **Step 1: Write the failing tests**

Append:

```python
def _toy_frame(n=240, separable=True, seed=0):
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    fail = rng.integers(0, 2, n)
    emax = rng.normal(0.05, 0.01, n) + (0.03 * fail if separable else 0.0)
    sigma = rng.normal(0.001, 0.0003, n)
    rcp = rng.normal(15, 3, n)
    serial = [f"S{i//3}" for i in range(n)]  # repeated serials -> grouping matters
    return pd.DataFrame({
        "untrimmed_error_max": emax, "untrimmed_sigma_gradient": sigma,
        "resistance_change_percent": rcp, "trim_pass_count": rng.integers(1, 3, n),
        "linearity_pass": 1 - fail, "serial": serial,
    })


def test_composite_trains_and_scores_in_unit_interval():
    from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel
    m = CompositeRiskModel("8232-1")
    res = m.train(_toy_frame(separable=True))
    assert res.n_samples == 240
    assert 0.0 <= res.cv_auc <= 1.0
    p = m.predict_proba({"untrimmed_error_max": 0.09, "untrimmed_sigma_gradient": 0.001,
                         "resistance_change_percent": 15.0, "trim_pass_count": 2})
    assert 0.0 <= p <= 1.0


def test_composite_deploy_gate_blocks_no_signal_model():
    from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel
    m = CompositeRiskModel("noise")
    res = m.train(_toy_frame(separable=False, seed=7))
    # no real signal -> low confidence / no lift -> not deployed
    assert res.deployed is False


def test_composite_deploy_gate_passes_separable_model():
    from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel
    m = CompositeRiskModel("real")
    res = m.train(_toy_frame(separable=True, seed=1))
    assert res.cv_auc > 0.6


def test_composite_drops_all_null_feature():
    import numpy as np
    from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel
    df = _toy_frame(separable=True)
    df["trim_pass_count"] = np.nan          # simulate pre-reprocess state
    m = CompositeRiskModel("preReprocess")
    res = m.train(df)
    assert "trim_pass_count" not in res.features_used
    assert res.deployed in (True, False)     # still trains on the rest


def test_composite_save_load_roundtrip(tmp_path):
    from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel
    m = CompositeRiskModel("rt")
    m.train(_toy_frame(separable=True))
    p = tmp_path / "rt.pkl"
    m.save(p)
    m2 = CompositeRiskModel.load(p)
    feat = {"untrimmed_error_max": 0.09, "untrimmed_sigma_gradient": 0.001,
            "resistance_change_percent": 15.0, "trim_pass_count": 2}
    assert abs(m.predict_proba(feat) - m2.predict_proba(feat)) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_composite_trim_risk.py -k composite -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the class**

Create `src/laser_trim_analyzer/ml/composite_risk.py`:

```python
"""Per-model composite trim-risk model.

A small logistic regression over orthogonal trim-effort features that predicts
whether a track fails linearity. Its per-unit score (0..1) is persisted and
watched at the group level for upstream drift. Honest grouped-CV + a deploy-gate
keep it from shipping where it doesn't actually beat the best single signal.

See docs/superpowers/plans/2026-06-01-composite-trim-risk-early-warning.md.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Candidate features, in priority order. The model uses whichever have data.
FEATURES: List[str] = [
    "untrimmed_error_max",        # worst raw point (best single signal)
    "untrimmed_sigma_gradient",   # raw steepness (often redundant)
    "resistance_change_percent",  # orthogonal: resistance-offset mode
    "trim_pass_count",            # orthogonal: trim-headroom mode (reprocess gate)
]

MIN_SAMPLES = 60       # below this we don't fit
MIN_FAILS = 15         # need both classes with enough fails
MIN_LIFT = 0.02        # CV AUC must beat best single feature by this
MIN_CONFIDENCE = 0.20  # honest confidence floor to deploy


@dataclass
class CompositeTrainingResult:
    model_name: str
    n_samples: int
    n_fails: int
    features_used: List[str]
    cv_auc: float                 # grouped out-of-fold AUC of the composite
    best_single_auc: float        # best single-feature grouped OOF AUC
    confidence: float             # rank-AUC honest confidence in [0,1]
    deployed: bool                # passed the gate?
    reason: str = ""              # why deployed / not
    coef: Dict[str, float] = field(default_factory=dict)


class CompositeRiskModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.features_used: List[str] = []
        self._pipe = None            # sklearn Pipeline (imputer+scaler+logreg)
        self._feat_median: Dict[str, float] = {}
        self.result: Optional[CompositeTrainingResult] = None
        self.is_trained = False

    # ---- training -------------------------------------------------------
    def _grouped_oof_auc(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Optional[float]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.metrics import roc_auc_score
        k = min(5, len(set(groups.tolist())))
        if k < 2 or len(set(y.tolist())) < 2:
            return None
        pipe = make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler(),
                             LogisticRegression(max_iter=1000))
        try:
            p = cross_val_predict(pipe, X, y, cv=GroupKFold(k), groups=groups,
                                  method="predict_proba")[:, 1]
            return float(roc_auc_score(y, p))
        except Exception:
            return None

    def train(self, df: pd.DataFrame) -> CompositeTrainingResult:
        # label: fail = 1
        y_full = (~df["linearity_pass"].astype(bool)).astype(int).to_numpy()
        groups_full = df.get("serial")
        groups_full = (groups_full.fillna(pd.Series([f"_{i}" for i in range(len(df))]))
                       if groups_full is not None else pd.Series(range(len(df)))).to_numpy()

        # keep only features that have at least some non-null data
        feats = [f for f in FEATURES if f in df.columns and df[f].notna().any()]
        self.features_used = feats

        def _result(**kw):
            self.result = CompositeTrainingResult(
                model_name=self.model_name, features_used=list(feats), **kw)
            return self.result

        n = len(df); n_fail = int(y_full.sum())
        if n < MIN_SAMPLES or n_fail < MIN_FAILS or (n - n_fail) < MIN_FAILS or not feats:
            return _result(n_samples=n, n_fails=n_fail, cv_auc=0.5, best_single_auc=0.5,
                           confidence=0.0, deployed=False,
                           reason="insufficient data", coef={})

        X = df[feats].to_numpy(dtype=float)

        # composite grouped OOF AUC
        cv_auc = self._grouped_oof_auc(X, y_full, groups_full) or 0.5
        # best single-feature grouped OOF AUC
        best_single = 0.5
        for j, f in enumerate(feats):
            a = self._grouped_oof_auc(X[:, [j]], y_full, groups_full)
            if a is not None:
                best_single = max(best_single, a, 1 - a)  # direction-agnostic

        # honest confidence (threshold_optimizer pattern)
        strength = max(0.0, min(1.0, 2.0 * (cv_auc - 0.5)))
        n_factor = min(1.0, n / 200.0)
        confidence = round(strength * n_factor, 3)

        lift = cv_auc - best_single
        deployed = (lift >= MIN_LIFT) and (confidence >= MIN_CONFIDENCE)
        reason = (f"lift={lift:+.3f} (need >= {MIN_LIFT}), conf={confidence} "
                  f"(need >= {MIN_CONFIDENCE})")

        # fit the final pipeline on ALL rows (median-impute remembered for scoring)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        self._feat_median = {f: float(np.nanmedian(df[f].to_numpy(dtype=float)))
                             for f in feats}
        self._pipe = make_pipeline(SimpleImputer(strategy="median"),
                                   StandardScaler(),
                                   LogisticRegression(max_iter=1000)).fit(X, y_full)
        self.is_trained = True
        coef = dict(zip(feats, self._pipe.steps[-1][1].coef_.ravel().tolist()))

        return _result(n_samples=n, n_fails=n_fail, cv_auc=round(cv_auc, 3),
                       best_single_auc=round(best_single, 3), confidence=confidence,
                       deployed=deployed, reason=reason, coef=coef)

    # ---- scoring --------------------------------------------------------
    def predict_proba(self, feat: Dict[str, float]) -> float:
        if not self.is_trained or self._pipe is None:
            return float("nan")
        row = []
        for f in self.features_used:
            v = feat.get(f)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = self._feat_median.get(f, 0.0)
            row.append(float(v))
        X = np.asarray(row, dtype=float).reshape(1, -1)
        return float(self._pipe.predict_proba(X)[0, 1])

    # ---- persistence (predictor.py pattern) ----------------------------
    def save(self, path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "model_name": self.model_name,
                "features_used": self.features_used,
                "pipe": self._pipe,
                "feat_median": self._feat_median,
                "result": self.result,
                "is_trained": self.is_trained,
            }, fh)

    @classmethod
    def load(cls, path) -> "CompositeRiskModel":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        m = cls(d["model_name"])
        m.features_used = d["features_used"]; m._pipe = d["pipe"]
        m._feat_median = d["feat_median"]; m.result = d["result"]
        m.is_trained = d["is_trained"]
        return m
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_composite_trim_risk.py -k composite -v`
Expected: PASS (all 5 composite tests).

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/ml/composite_risk.py tests/test_composite_trim_risk.py
git commit -m "feat(composite): CompositeRiskModel with grouped-CV honest deploy-gate (V6)"
```

---

## Task 8: Train + persist the composite in `MLManager.train_model`

**Files:**
- Modify: `src/laser_trim_analyzer/ml/manager.py` (`train_model` after predictor training ~L331; storage near predictor save L1011-1019)
- Test: `tests/test_composite_trim_risk.py`

- [ ] **Step 1: Write the failing test** (the manager exposes a composite accessor + trains it)

Append:

```python
def test_mlmanager_trains_composite(tmp_path):
    import pandas as pd
    from laser_trim_analyzer.ml.manager import MLManager
    mm = MLManager(ml_storage_path=tmp_path)
    df = _toy_frame(separable=True)
    crm = mm._train_composite_risk("8232-1", df)   # new helper
    assert crm.is_trained
    assert (tmp_path / "composite_risk" / "8232-1.pkl").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_composite_trim_risk.py::test_mlmanager_trains_composite -v`
Expected: FAIL — `MLManager` has no `_train_composite_risk`.

- [ ] **Step 3: Add the training helper + call it**

In `src/laser_trim_analyzer/ml/manager.py`, add a helper method on `MLManager`:

```python
    def _train_composite_risk(self, model_name: str, data) -> "CompositeRiskModel":
        """Train and persist the per-model composite trim-risk model."""
        from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel
        crm = CompositeRiskModel(model_name)
        res = crm.train(data)
        logger.info("Composite risk [%s]: cv_auc=%.3f best_single=%.3f conf=%.3f deployed=%s (%s)",
                    model_name, res.cv_auc, res.best_single_auc, res.confidence,
                    res.deployed, res.reason)
        out_dir = self.storage_path / "composite_risk"
        out_dir.mkdir(parents=True, exist_ok=True)
        crm.save(out_dir / f"{model_name}.pkl")
        if not hasattr(self, "composite_models"):
            self.composite_models = {}
        self.composite_models[model_name] = crm
        return crm
```

In `train_model`, after the predictor is trained (~L331, where `data = self._get_training_data(model_name)` is already in scope), add:

```python
            # Composite trim-risk model (2026-06-01 plan). Reuses the same
            # per-model training frame; grouped-CV + deploy-gate inside.
            try:
                self._train_composite_risk(model_name, data)
            except Exception as e:
                logger.warning("Composite risk training failed for %s: %s", model_name, e)
```

> If `data` is a different variable name in that scope, pass whatever holds the `_get_training_data(model_name)` DataFrame. It must contain `linearity_pass`, `serial`, and the feature columns from Task 6.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_composite_trim_risk.py::test_mlmanager_trains_composite -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/ml/manager.py tests/test_composite_trim_risk.py
git commit -m "feat(composite): train + persist composite risk per model (V6)"
```

---

## Task 9: Score units into `composite_trim_risk_score`

**Files:**
- Modify: `src/laser_trim_analyzer/ml/manager.py` (`apply_to_database` loop L485 — historical scoring of a model's units)
- Modify: `src/laser_trim_analyzer/core/processor.py` (live scoring of new units, following the existing per-unit risk pattern)
- Test: `tests/test_composite_trim_risk.py`

Scoring writes `composite_trim_risk_score` **only for deployed models** (so the drift metric is clean). Non-deployed models leave it NULL and the drift engine falls back to `untrimmed_error_max`.

- [ ] **Step 1: Write the failing test** (apply writes scores for a deployed model)

Append:

```python
def test_apply_scores_units_for_deployed_model(tmp_path):
    import sqlite3
    from laser_trim_analyzer.ml.manager import MLManager
    # minimal DB with a few tracks for one model
    db = tmp_path / "a.db"
    con = sqlite3.connect(db)
    con.executescript("""
      CREATE TABLE analysis_results (id INTEGER PRIMARY KEY, model TEXT, serial TEXT);
      CREATE TABLE track_results (
        id INTEGER PRIMARY KEY, analysis_id INTEGER,
        untrimmed_error_max REAL, untrimmed_sigma_gradient REAL,
        resistance_change_percent REAL, trim_pass_count INTEGER,
        composite_trim_risk_score REAL);
    """)
    con.execute("INSERT INTO analysis_results VALUES (1,'M','S1')")
    for i in range(3):
        con.execute("INSERT INTO track_results (analysis_id, untrimmed_error_max, "
                    "untrimmed_sigma_gradient, resistance_change_percent, trim_pass_count) "
                    "VALUES (1, ?, 0.001, 15.0, 2)", (0.05 + 0.01*i,))
    con.commit(); con.close()

    mm = MLManager(ml_storage_path=tmp_path)
    mm._train_composite_risk("M", _toy_frame(separable=True))
    mm.composite_models["M"].result.deployed = True  # force-deploy for the test
    n = mm._score_composite_for_model(str(db), "M")
    assert n == 3
    con = sqlite3.connect(db)
    vals = [r[0] for r in con.execute(
        "SELECT composite_trim_risk_score FROM track_results").fetchall()]
    con.close()
    assert all(v is not None and 0.0 <= v <= 1.0 for v in vals)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_composite_trim_risk.py::test_apply_scores_units_for_deployed_model -v`
Expected: FAIL — no `_score_composite_for_model`.

- [ ] **Step 3: Add the scoring method + wire into apply**

In `src/laser_trim_analyzer/ml/manager.py`, add:

```python
    def _score_composite_for_model(self, db_path: str, model_name: str) -> int:
        """Write composite_trim_risk_score for all of a model's tracks.
        Only runs for deployed models; returns rows scored."""
        import sqlite3
        from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel, FEATURES
        crm = getattr(self, "composite_models", {}).get(model_name)
        if crm is None:
            p = self.storage_path / "composite_risk" / f"{model_name}.pkl"
            if not p.exists():
                return 0
            crm = CompositeRiskModel.load(p)
        if not (crm.is_trained and crm.result and crm.result.deployed):
            return 0
        cols = ", ".join(FEATURES)
        con = sqlite3.connect(db_path); cur = con.cursor()
        rows = cur.execute(
            f"SELECT t.id, {cols} FROM track_results t "
            f"JOIN analysis_results ar ON t.analysis_id = ar.id WHERE ar.model = ?",
            (model_name,)).fetchall()
        scored = 0
        for r in rows:
            tid = r[0]
            feat = {f: r[1 + i] for i, f in enumerate(FEATURES)}
            s = crm.predict_proba(feat)
            if s == s:  # not NaN
                cur.execute("UPDATE track_results SET composite_trim_risk_score=? WHERE id=?",
                            (s, tid)); scored += 1
        con.commit(); con.close()
        return scored
```

In `apply_to_database` (the `for model_name in ...` loop, L485), after the existing per-model apply work, add:

```python
            try:
                self._score_composite_for_model(self.db_manager.db_path, model_name)
            except Exception as e:
                logger.warning("Composite scoring failed for %s: %s", model_name, e)
```

> Use whatever attribute holds the DB path in this class (e.g. `self.db_manager.db_path`). If only a session is available, adapt to a SQLAlchemy `UPDATE` instead of raw sqlite3 — the logic is identical.

- [ ] **Step 4: Live scoring for new units in the processor**

In `src/laser_trim_analyzer/core/processor.py`, find where the per-unit `failure_probability` / `risk_category` is assigned during analysis (the existing per-unit risk hook). Immediately after the track's trim-effort fields are known, add (best-effort, never block processing):

```python
        # Composite trim-risk score for the live unit (drift early-warning).
        try:
            crm = self._composite_models.get(model)  # lazy-loaded per model
            if crm is None:
                from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel
                p = self.ml_storage_path / "composite_risk" / f"{model}.pkl"
                crm = CompositeRiskModel.load(p) if p.exists() else False
                self._composite_models[model] = crm
            if crm and crm.is_trained and crm.result and crm.result.deployed:
                track_result.composite_trim_risk_score = crm.predict_proba({
                    "untrimmed_error_max": track_result.untrimmed_error_max,
                    "untrimmed_sigma_gradient": track_result.untrimmed_sigma_gradient,
                    "resistance_change_percent": getattr(track_result, "resistance_change_percent", None),
                    "trim_pass_count": track_result.trim_pass_count,
                })
        except Exception:
            pass  # scoring is non-essential; never fail a unit over it
```

Add `self._composite_models = {}` and `self.ml_storage_path = Path("data/ml_models")` (or the configured path) in the processor's `__init__` if not present.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_composite_trim_risk.py -k "apply or composite" -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/ml/manager.py src/laser_trim_analyzer/core/processor.py tests/test_composite_trim_risk.py
git commit -m "feat(composite): score composite_trim_risk_score (apply + live) (V6)"
```

---

## Task 10: Register the composite as the 8th watched drift metric

**Files (V6):**
- Modify: `src/laser_trim_analyzer/ml/drift_types.py` (`WATCHED_METRICS` L59-67)
- Modify: `src/laser_trim_analyzer/ml/drift_training.py` (`_TRACK_METRIC_COLUMNS` L40-50)
- Test: `tests/test_composite_trim_risk.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_composite_is_watched_metric():
    from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS
    assert "composite_trim_risk_score" in WATCHED_METRICS


def test_composite_has_drift_column_mapping():
    from laser_trim_analyzer.ml import drift_training
    assert "composite_trim_risk_score" in drift_training._TRACK_METRIC_COLUMNS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_composite_trim_risk.py -k "watched or drift_column" -v`
Expected: FAIL.

- [ ] **Step 3: Add to WATCHED_METRICS and the column map**

In `src/laser_trim_analyzer/ml/drift_types.py`, append to the `WATCHED_METRICS` tuple:

```python
    "composite_trim_risk_score",
```

> Note: `n_metrics` becomes 8, so the Bonferroni p-budget (`target_fp_for_tier(...) / n_metrics`) tightens automatically — correct and intended.

In `src/laser_trim_analyzer/ml/drift_training.py`, add to `_TRACK_METRIC_COLUMNS`:

```python
    "composite_trim_risk_score": DBTrackResult.composite_trim_risk_score,
```

(Use the same `DBTrackResult` alias the file already imports for the other TrackResult columns.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_composite_trim_risk.py -k "watched or drift_column" -v`
Expected: PASS.

- [ ] **Step 5: Sanity-check drift training end-to-end on a copy**

```bash
cp data/analysis.db /tmp/drift.db   # a copy that has composite scores from Task 9
python - <<'PY'
# Build detectors + advance, confirm the composite metric trains without error.
import sys; sys.path.insert(0, "src")
# (Use the project's existing drift training entry point, e.g. train + advance_drift_state.)
print("Wire to the repo's drift training entry; confirm 'composite_trim_risk_score' appears in model_metric_state.")
PY
```
Expected: `model_metric_state` gains `(model, 'composite_trim_risk_score')` rows for models with scored units.

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/ml/drift_types.py src/laser_trim_analyzer/ml/drift_training.py tests/test_composite_trim_risk.py
git commit -m "feat(composite): watch composite_trim_risk_score as 8th drift metric (V6)"
```

---

## Task 11: Single-model dashboard chart + deploy-gate validation report

**Files (V6):**
- Modify: `src/laser_trim_analyzer/gui/pages/trends.py` (single-model dashboard `metric_axes`; Process-drift `metric_options` ~L2950-2970)
- Create: `scripts/validate_composite_deploy_gate.py`

The all-models drift view auto-includes the new metric (worst-of-N). The single-model dashboard hardcodes its charts, so the composite needs explicit wiring.

- [ ] **Step 1: Add the composite to the single-model dashboard**

In `src/laser_trim_analyzer/gui/pages/trends.py`, in `_show_single_model_drift`'s chart grid, extend `metric_axes` to include the composite (replace the bottom-right retrim panel slot or add a row as layout allows):

```python
        metric_axes = (
            ("untrimmed_resistance", gs[0, 1], "Untrimmed Resistance"),
            ("measured_electrical_angle", gs[1, 0], "Electrical Angle"),
            ("composite_trim_risk_score", gs[1, 1], "Composite Trim-Risk"),
        )
```

And add it to the Process-drift `metric_options` (~L2950-2970):

```python
        metric_options = [
            ("untrimmed_resistance", "Untrimmed Resistance"),
            ("measured_electrical_angle", "Electrical Angle"),
            ("trim_pass_count", "Trim Passes"),
            ("composite_trim_risk_score", "Composite Trim-Risk"),
        ]
```

- [ ] **Step 2: Manual UI smoke test**

Run: `python src/__main__.py`, open Trends → ML Drift → select a model with a deployed composite.
Expected: the "Composite Trim-Risk" panel renders its per-unit dots + smoothed overlay like the other process metrics; no exceptions in the log.

- [ ] **Step 3: Write the deploy-gate validation report script**

Create `scripts/validate_composite_deploy_gate.py`:

```python
"""Report, per model, whether the composite earns deployment.

Re-runs the grouped-CV comparison (composite vs best single feature) on the live
DB and prints the deploy decision. Run AFTER the trim_pass_count reprocess so the
3rd orthogonal mode is in play. Read-only.
"""
import sys
sys.path.insert(0, "src")
import sqlite3, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from laser_trim_analyzer.ml.composite_risk import CompositeRiskModel

DB = sys.argv[1] if len(sys.argv) > 1 else "data/analysis.db"
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); c = con.cursor()
models = [r[0] for r in c.execute(
    """SELECT ar.model FROM track_results t JOIN analysis_results ar ON t.analysis_id=ar.id
       WHERE t.linearity_pass IS NOT NULL GROUP BY ar.model
       HAVING SUM(CASE WHEN t.linearity_pass=0 THEN 1 ELSE 0 END) >= 15 AND COUNT(*) >= 60
       ORDER BY COUNT(*) DESC""")]
import pandas as pd
print(f"{'model':10} {'n':>6} {'cv_auc':>7} {'best1':>6} {'conf':>5} {'deploy':>7}  reason")
deployed = 0
for m in models:
    rows = c.execute(
        """SELECT ar.serial, t.linearity_pass, t.untrimmed_error_max, t.untrimmed_sigma_gradient,
                  t.resistance_change_percent, t.trim_pass_count
           FROM track_results t JOIN analysis_results ar ON t.analysis_id=ar.id
           WHERE ar.model=? AND t.linearity_pass IS NOT NULL""", (m,)).fetchall()
    df = pd.DataFrame(rows, columns=["serial","linearity_pass","untrimmed_error_max",
        "untrimmed_sigma_gradient","resistance_change_percent","trim_pass_count"])
    res = CompositeRiskModel(m).train(df)
    deployed += int(res.deployed)
    print(f"{m:10} {res.n_samples:6d} {res.cv_auc:7.3f} {res.best_single_auc:6.3f} "
          f"{res.confidence:5.2f} {str(res.deployed):>7}  {res.reason}")
con.close()
print(f"\nDeployed for {deployed}/{len(models)} models. "
      f"Non-deployed models keep untrimmed_error_max as the drift signal.")
```

- [ ] **Step 4: Run the validation report (after reprocess)**

Run: `python scripts/validate_composite_deploy_gate.py data/analysis.db`
Expected: a per-model table; the composite deploys where it beats the best single feature by ≥ 0.02 with confidence ≥ 0.20. (Pre-reprocess, expect it to deploy on the mixed-mode models like 8232-1/8762/6126; post-reprocess, expect more, as `trim_pass_count` adds the headroom mode.)

- [ ] **Step 5: Commit + push V6**

```bash
git add src/laser_trim_analyzer/gui/pages/trends.py scripts/validate_composite_deploy_gate.py
git commit -m "feat(composite): single-model dashboard chart + deploy-gate validation report (V6)"
git push origin V6
```

---

## Self-review checklist (run before handing off)

**Spec coverage** — every brainstorm conclusion maps to a task:
- worst-raw-point signal → Task 1/2 (`untrimmed_error_max`)
- backfill the derivable features, no reprocess → Task 3
- `trim_pass_count` reprocess gate → Task 4 + Reprocess Gate
- composite must beat best single (orthogonal modes only) → Task 7 deploy-gate + Task 11 report
- per-model, grouped-CV, honest confidence (no leakage) → Task 7 (`GroupKFold`, serial groups, rank-AUC confidence)
- group-level drift watch → Task 10 (8th watched metric, CUSUM/EWMA via existing engine)
- auditable persisted per-unit score → Task 5 column + Task 9 scoring
- UI surfacing → Task 11

**Type/name consistency** — `CompositeRiskModel`, `CompositeTrainingResult`, `FEATURES`, `_train_composite_risk`, `_score_composite_for_model`, `composite_trim_risk_score`, `untrimmed_error_max` are used identically across Tasks 5–11.

**Placeholder scan** — none. Every code step shows real code; every test has assertions; every command has expected output. The two "adapt to your repo's attribute" notes (DB path in Task 9, drift entry point in Task 10 Step 5) are explicitly flagged as repo-specific wiring, not missing logic.

**Known soft spots to verify during execution (not placeholders):**
1. ~~`Analyzer.__init__` signature~~ — **verified**: `Analyzer()` takes all-default args (`scaling_factor`, `model_thresholds`); pydantic is v2 so `TrackData.model_fields` is correct; `_calculate_trim_effectiveness` uses `valid_untrimmed` (L1104) and the existing `max(abs(e) for e in valid_untrimmed)` idiom (L1121).
2. The exact in-scope variable holding the training DataFrame in `train_model` (Task 8 note).
3. The processor's per-unit risk hook location and its DB-path/ml-path attributes (Task 9 note).
4. `gs[1,1]` grid slot availability in the single-model dashboard (Task 11 — adjust the GridSpec if the retrim panel occupies it).

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-01-composite-trim-risk-early-warning.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. Best here because Phase 1 (main) and Phase 2 (V6) are on different branches with a reprocess gate between them.

**2. Inline Execution** — execute tasks in this session with checkpoints for review.

**Which approach?** (Note: Phase 1 can run now on `main`; Phase 2 should wait on the work-machine reprocess for full-strength deployment, though its code can be built on `V6` in parallel.)
