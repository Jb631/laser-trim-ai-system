# Spec 2 — Multi-Metric Drift Detector Design

**Date:** 2026-05-30
**Status:** Approved, ready for implementation plan
**Parent specs:**
- `docs/superpowers/specs/2026-05-14-app-mission-realignment-design.md` (anchor)
- `docs/superpowers/specs/2026-05-30-app-redesign-design.md` (V6 umbrella)
**Depends on:** Spec 1 (`untrimmed_sigma_gradient` column) — merged into V6 via `f25f600`.
**Target branch:** `V6` only. Main remains the V5 production deployment.
**Source:** Brainstorming session with James 2026-05-30 (after Spec 1 implementation landed). Captures the multi-metric drift detector design that Spec 3's Triage and Model pages will consume.

This spec describes the **drift detection subsystem** that watches all 8 watched metrics per model, classifies alerts as step-change vs slow-drift, and tightens false-positive rates via ML-learned per-model per-metric sensitivity. It is purely a backend / data-layer spec — no UI changes. Spec 3 owns all UI surfaces.

---

## Mission alignment

From the V6 umbrella spec:

> *Multi-metric drift detector. Step-change vs slow-drift classifier. Per-tier per-metric threshold tightening with config persisted to disk. Smoothness signal feeding alerts.*

Spec 2 directly attacks the anchor doc's "flags everything" complaint via three converging fixes:

1. **Multi-metric watch** — alerts include which metric drifted, so the user can act on the right upstream signal instead of guessing from the single post-trim sigma.
2. **Step-change vs slow-drift classification** — the user knows whether a lot just shifted (act today) or the model is slowly degrading (emerging concern).
3. **ML-learned per-model sensitivity** — each model's CUSUM/EWMA/step-change thresholds are picked so the model's own historical data would have generated only a target false-positive rate. No more global "one threshold, all models" mismatch.

A single user-facing **sensitivity preset** (`loose` / `standard` / `tight` / `strict`) sets the target FP rate for each tier; all per-model per-metric thresholds derive from that preset + each model's own baseline.

---

## Architecture

### Class structure

Replace the current single-metric `ModelDriftDetector` (one model, sigma_gradient only) with a two-level structure:

```
MetricDetector (per model, per metric)              ← new class
├── baseline_mean, baseline_std, baseline_count
├── h_warning, h_drift, h_oc                       (CUSUM thresholds)
├── L_warning, L_drift, L_oc                       (EWMA control-limit widths)
├── z_warning, z_drift, z_oc                       (step-change z-thresholds)
├── ewma_state, cusum_pos, cusum_neg               (live runtime state)
├── is_trained
├── update(value) -> MetricStatus
├── get_status() -> MetricStatus
└── reset_runtime()

ModelDriftDetector (per model)                     ← refactored
├── model: str
├── metrics: Dict[str, MetricDetector]              (8 entries)
├── update(sample: Dict[str, float]) -> ModelDriftStatus
└── get_status() -> ModelDriftStatus
```

The old `ModelDriftDetector` class is retained alongside the new one until Spec 3 retires the UI that consumes it. Spec 2 lands new code; Spec 3 removes the old.

### The 8 watched metrics

| # | Metric | Data source | Type | Notes |
|---|---|---|---|---|
| 1 | `sigma_gradient` | `TrackResult.sigma_gradient` | post-trim curve smoothness | existing |
| 2 | `untrimmed_sigma_gradient` | `TrackResult.untrimmed_sigma_gradient` | upstream element-quality | Spec 1 |
| 3 | `untrimmed_resistance` | `TrackResult.untrimmed_resistance` | starting resistance | existing |
| 4 | `linearity_error` | `TrackResult.linearity_error` (or `final_linearity_error_shifted`) | post-trim error | existing; implementer picks correct column |
| 5 | `measured_electrical_angle` | `TrackResult.measured_electrical_angle` | element angle drift | existing |
| 6 | `trim_pass_count` | `TrackResult.trim_pass_count` | laser passes needed | existing; integer-valued but treated as float for sigma math |
| 7 | `resistance_change_percent` | `TrackResult.resistance_change_percent` | material removed | existing |
| 8 | `max_smoothness_value` | `SmoothnessResult.max_smoothness_value` | worst-case smoothness per test | only metric with a separate data source |

All 8 flow through the same `MetricDetector` machinery. Smoothness's separate data source affects only the query path during training; detection logic is uniform.

---

## Sensitivity learning

### Preset → target FP rate per tier

| Preset | Warning | Drift | Out-of-Control |
|---|---|---|---|
| `loose` | 10 % | 5 % | 1 % |
| `standard` (default) | 5 % | 1 % | 0.1 % |
| `tight` | 1 % | 0.1 % | 0.01 % |
| `strict` | 0.1 % | 0.01 % | 0.001 % |

The preset is a single string in `~/.laser_trim_analyzer/config.yaml`; default `standard`; Spec 3's Settings UI reads and writes it.

### Threshold math (per model, per metric, per tier)

Given baseline `(μ, σ, n)` for one (model, metric) pair, and a target false-positive rate `p` for one tier:

- **EWMA control-limit width:** `L = Φ⁻¹(1 − p/2)` (two-sided). Lambda fixed at 0.2. EWMA process std = `σ × √(λ / (2 − λ))`.
- **Step-change z-threshold:** `z = Φ⁻¹(1 − p)` (one-sided each direction; the comparison is `|recent_mean − μ| × √n / σ`).
- **CUSUM decision threshold:** SPC approximation `h ≈ −ln(p) / k_cusum` with `k_cusum = 0.5 σ`. Verify against ARL targets after the first real-data training run; refine the constant if production data shows the approximation is biased.

All three are derived deterministically from `(σ, p)` — no optimization step, no convergence loop. Computed once during training, cached in the DB row.

### Three independent checks per metric per sample

Each `MetricDetector.update(value)` runs three checks against the current sample:

| Check | Watches | Trips when |
|---|---|---|
| CUSUM (slow drift) | cumulative deviation from baseline mean | `cusum_pos > h` or `cusum_neg < −h` |
| EWMA (slow drift) | exponentially-weighted moving average | `|ewma_state − μ| > L × σ_EWMA` |
| Step-change (lot shift) | last N samples' mean vs baseline | `|recent_mean − μ| × √N / σ > z` |

`N` (step-change window) = 5 samples. Fixed in v1; not user-tunable. Configurable as a code constant.

When multiple checks trip on the same sample at different tier severities, the **highest-severity tier wins** (CUSUM at Drift level + EWMA at Warning level → metric reports Drift).

When step-change and slow-drift both trip on the same sample at the same severity, **step-change wins** for the displayed alert type. It's the more actionable form (a lot just shifted today, vs. a 14-day trend).

### Insufficient-history fallback

A `(model, metric)` row with `baseline_count < 30` is written with `is_trained=False` and `tier=Stable` permanently. The detector still accumulates runtime state, so once a model accumulates samples, the next training run flips `is_trained=True` and the thresholds activate. New models silently warm up.

---

## Tier resolution

A model's **overall tier** is the worst-of across its 8 metric tiers:

`Stable < Warning < Drift < OutOfControl`

The Triage page (Spec 3) displays the model's overall tier; the Model page displays per-metric tiers in the glance row.

The model's **alert type** (step-change vs slow-drift) is taken from whichever metric drove the overall tier — and within that metric, step-change wins over slow-drift per the rule above.

---

## Storage

### New table — `model_metric_state`

```sql
CREATE TABLE model_metric_state (
    id INTEGER PRIMARY KEY,
    model VARCHAR(50) NOT NULL,
    metric VARCHAR(50) NOT NULL,             -- one of the 8 metric names
    baseline_cutoff_date DATETIME,
    baseline_mean FLOAT,
    baseline_std FLOAT,
    baseline_count INTEGER NOT NULL,
    is_trained BOOLEAN NOT NULL DEFAULT 0,
    -- thresholds per tier (computed from preset + baseline)
    h_warning FLOAT, h_drift FLOAT, h_oc FLOAT,
    L_warning FLOAT, L_drift FLOAT, L_oc FLOAT,
    z_warning FLOAT, z_drift FLOAT, z_oc FLOAT,
    -- live runtime state, updated on each new sample
    ewma_state FLOAT,
    cusum_pos FLOAT NOT NULL DEFAULT 0,
    cusum_neg FLOAT NOT NULL DEFAULT 0,
    last_updated DATETIME,
    UNIQUE (model, metric)
);
CREATE INDEX idx_model_metric_state_model ON model_metric_state (model);
CREATE INDEX idx_model_metric_state_metric ON model_metric_state (metric);
```

About 320 stored numbers per model (40 per metric × 8 metrics). Indexed for the two predominant query patterns: "all metrics for one model" (Model page) and "one metric across all models" (training and preview).

### Existing `model_ml_state` table — untouched

The current single-metric system reads/writes `sigma_gradient` baseline and CUSUM state in `model_ml_state`. Spec 2 reads `drift_baseline_cutoff_date` from there as input but writes nothing back. The old state stays as a historical record. When Spec 3 retires the old `ModelDriftDetector` class, the table's drift-related columns become unused; deletion deferred to a future cleanup.

### Sensitivity preset — `~/.laser_trim_analyzer/config.yaml`

```yaml
drift:
  sensitivity: standard   # one of: loose, standard, tight, strict
```

Default `standard`. Spec 3's Settings UI is the only writer.

---

## Public API

```python
# laser_trim_analyzer/ml/drift_detector.py (V6)

class MetricDetector:
    """One model × one metric. Holds baseline, thresholds, runtime state."""

    def update(self, value: float) -> MetricStatus: ...
    def get_status(self) -> MetricStatus: ...
    def reset_runtime(self) -> None: ...

class ModelDriftDetector:
    """Container for one model's 8 MetricDetector instances."""

    def __init__(self, model: str, metrics: Dict[str, MetricDetector]): ...
    def update(self, sample: Dict[str, float]) -> ModelDriftStatus: ...
    def get_status(self) -> ModelDriftStatus: ...


# laser_trim_analyzer/ml/manager.py (V6)

def get_drifting_models(
    db: DatabaseManager,
    sensitivity_preset: str = "standard",
) -> List[ModelAlertSummary]:
    """For Triage page. Sorted list of currently-flagged models, worst first.
    Returns empty list when nothing is above Stable."""

def get_model_drift_status(db: DatabaseManager, model: str) -> ModelDriftStatus:
    """For Model page. Full per-metric breakdown for one model."""

def train_drift_detector(
    db: DatabaseManager,
    sensitivity_preset: str,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> TrainingSummary:
    """Force retrain. Called by Settings 'Retrain' button + first-startup
    auto-train.  progress_callback signature: (current_model, done, total)."""

def preview_alert_count(
    db: DatabaseManager,
    sensitivity_preset: str,
) -> Dict[str, int]:
    """For Settings sensitivity-preset live preview.  Returns count per tier
    (e.g. {'warning': 12, 'drift': 3, 'out_of_control': 0}) that would flag
    under the candidate preset using current cached runtime state."""
```

Data shapes:

```python
@dataclass
class MetricStatus:
    metric: str
    tier: DriftTier
    alert_type: Optional[AlertType]  # StepChange / SlowDrift / None
    magnitude: float                 # σ-units over threshold, for sort/display
    baseline_mean: float
    baseline_std: float
    recent_mean: Optional[float]
    recent_count: int
    is_trained: bool

@dataclass
class ModelDriftStatus:
    model: str
    overall_tier: DriftTier
    worst_metric: Optional[str]
    worst_alert_type: Optional[AlertType]
    per_metric: Dict[str, MetricStatus]
    last_processed: datetime

@dataclass
class ModelAlertSummary:
    model: str
    tier: DriftTier
    alert_type: AlertType
    worst_metric: str
    magnitude: float

@dataclass
class TrainingSummary:
    models_trained: int
    metrics_per_model: int        # 8
    skipped_insufficient_data: List[Tuple[str, str]]
    duration_seconds: float
```

The processor pipeline (file ingest) calls `ModelDriftDetector(model).update(new_sample)` on every new record; the detector persists runtime state synchronously so subsequent `get_drifting_models` / `get_model_drift_status` reads from the same DB row see the new state immediately. No separate cache layer.

---

## Training trigger + migration

### Migration (V6 first startup)

1. Idempotent `CREATE TABLE IF NOT EXISTS model_metric_state` plus its two indexes.
2. If the table is empty after the create, log `"First-run drift training: N models × 8 metrics"` and trigger `train_drift_detector` in a background thread.
3. Subsequent startups: table is populated, no auto-retrain.

### Manual retrain

Spec 3's Settings UI exposes a `Retrain drift detector` button. Calls `train_drift_detector(db, preset, progress_callback)` in a background thread. Progress shown in the UI.

### Sensitivity preset change

When the Settings preset slider moves to a new value, `train_drift_detector` is **not** automatically rerun. Instead, threshold values in existing `model_metric_state` rows are **recomputed in place** (cheap — same baseline, just different target FP rates → different `h`/`L`/`z`). Runtime state (cusum, ewma) is preserved. This is fast (< 1 second) and makes the slider feel instant.

The `preview_alert_count` function is used by the UI to show "would flag" counts as the slider moves, before committing the change.

---

## Testing

| Layer | Tests |
|---|---|
| Inverse-CDF math | Given p=0.05, p=0.01, p=0.001, verify L ≈ 1.96 / 2.58 / 3.29 and z ≈ 1.645 / 2.326 / 3.090 to within 0.01. |
| MetricDetector — flat input | A series of samples at the baseline mean never trips any check. Synthetic, no DB. |
| MetricDetector — injected step | Switch from mean=0 series to mean=3σ series after sample 30; verify step-change fires at the right severity. |
| MetricDetector — slow ramp | Linear ramp from mean=0 to mean=2σ over 50 samples; verify CUSUM and EWMA fire, step-change does not. |
| MetricDetector — all-NaN | Pass NaN samples; verify no crash, runtime state unchanged. |
| ModelDriftDetector — worst-of | Two metrics, one Warning, one Drift; verify overall_tier=Drift, worst_metric is the Drift one. |
| ModelDriftDetector — alert-type priority | Step-change at Warning + slow-drift at Drift; verify the SlowDrift tier wins (higher severity) but the displayed alert type is the higher-severity metric's. |
| Training | Fixture DB with 5 models × 1 metric × 50 historical samples. Verify (1) 5 rows written, (2) baseline matches numpy `mean`/`std` of the input, (3) thresholds match the FP-rate math, (4) running twice doesn't double-write. |
| Training — insufficient history | Fixture with 20 historical samples (< 30); verify row written with `is_trained=False`, `tier=Stable`. |
| `get_drifting_models` | Fixture with mixed-tier models; verify sorted by severity desc, only flagged returned, empty list when nothing flagged. |
| `get_model_drift_status` | Fixture; verify all 8 metric records returned, `recent_mean` populated correctly. |
| `preview_alert_count` | Fixture; switch preset between `loose` and `tight`, verify count changes monotonically (tight should never have more alerts than loose). |
| Migration | Empty DB → first-startup creates table + triggers training. Second startup: no re-create, no re-train. |
| Smoothness path | Smoothness metric reads from `SmoothnessResult`, not `TrackResult`. Verify training picks up smoothness data and stores baseline correctly. |
| Smoke regression | Existing `tests/test_drift_dashboard_*.py` still pass against V6 (the old `ModelDriftDetector` class stays until Spec 3 retires it). |

---

## Non-goals (explicit out-of-scope)

- **No UI changes.** Triage page, Model page glance row, Settings threshold-tuning UI, "Retrain drift detector" button — all owned by Spec 3.
- **No retirement of the old `ModelDriftDetector` class.** Stays until Spec 3 ports the UI off it. Spec 2 lands new code beside it.
- **No per-model sensitivity overrides.** Single global preset.
- **No Western Electric rules 2–8.** Rule 1 only (single point outside control limits).
- **No backfill of historical alerts.** Past data is for training only.
- **No real-time streaming.** Batch detection on file processing.
- **No automatic periodic retraining.** Manual button + auto on first V6 startup only.
- **No threshold persistence in user config YAML beyond the single sensitivity preset.** Per-metric per-tier thresholds derive from preset + per-model baseline; not separately configurable.
- **No changes to main / V5.** V6 branch only.
- **No Excel export of drift alerts** in this spec; if desired, follow-up.

---

## Deferred to implementation phase (not blocking)

- **CUSUM `h` SPC approximation calibration.** Use `h ≈ −ln(p) / k_cusum` with `k_cusum = 0.5σ` in v1. Verify against ARL targets after first real-data training run; refine the constant if production data shows it's biased.
- **First-N-stable-records default.** For models without an existing `baseline_cutoff_date`, take the first 30 records by `file_date` order. Accept some baseline contamination in v1; consider median+MAD if it becomes a problem.
- **Step-change window size.** Fixed at N=5 samples in v1. Promote to a constant in `utils/constants.py` so it can be tuned without touching detector code.
- **Background-training thread management.** Use the existing `get_thread_manager()` pattern from the file-scan flow (see `gui/pages/process.py:399`). The progress callback is the only interaction surface.
- **Concurrency around `model_metric_state` updates.** Use the existing `DatabaseManager._write_lock` since SQLite serializes writes anyway; no per-metric finer lock needed.

---

## Implementation sequencing (informational — owned by the implementation plan)

Spec 2 decomposes naturally into ~6-7 plan tasks:

1. New ORM model `ModelMetricState` + idempotent startup migration.
2. New dataclasses (`MetricStatus`, `ModelDriftStatus`, `ModelAlertSummary`, `TrainingSummary`) and enums (`DriftTier`, `AlertType`).
3. `MetricDetector` class with the three checks (CUSUM, EWMA, step-change) and threshold math.
4. `ModelDriftDetector` container + worst-of aggregation.
5. `train_drift_detector` orchestration + first-startup auto-trigger.
6. `get_drifting_models`, `get_model_drift_status`, `preview_alert_count` public functions in `ml/manager.py`.
7. Tests at each layer per the testing matrix above.

The implementation plan (`writing-plans` skill output) elaborates each into TDD-style steps.
