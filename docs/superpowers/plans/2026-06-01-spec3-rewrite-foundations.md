# Spec 3 — Implementation Foundations & Corrections (V6 UI)

**Date:** 2026-06-01
**Status:** Authoritative. The rewritten Spec 3a–3e plans build on this document.
**Supersedes:** the shared-contract sections of the 2026-05-30 spec3a–3e plans.
**Parent specs:**
- `docs/superpowers/specs/2026-05-14-app-mission-realignment-design.md` (anchor — the mission)
- `docs/superpowers/specs/2026-05-30-app-redesign-design.md` (V6 umbrella)
- `docs/superpowers/specs/2026-05-30-spec3-ui-shell-design.md` (Spec 3 umbrella)

This is the single source of truth for everything the five Spec-3 sub-plans share:
verified API signatures, the mission/QA/manufacturing correctness rules every page
must obey, the corrected cross-cutting contracts (PageBase, V6App, theme, threading,
test infra), the two new data-layer helpers, and the decision log for the structural
changes (deferred V5 retirement, real Settings ports, auto-train opt-out).

**Why this exists.** A 2026-06-01 senior review of the original spec3a–3e plans found
7 critical and ~10 important defects. The root cause was that the plans assumed APIs
that don't exist or differ from reality (`AnalysisStatus.SKIPPED`, `ProcessingStatus.current_file_index`,
`Processor(db=...)`, widget reparenting via `widget.master=`, a metric→column identity
that silently plots the wrong signal). Every signature below was verified against the
committed code on the `V6` branch on 2026-06-01. **Implementers: trust this document over
memory or the original plans.**

---

## 0. The mission (keep this in front of you)

> *Which product models' element production is drifting (sudden lot shift or slow
> trend), how bad, and what evidence do I take to the engineers and MEs?*

Three jobs, nothing else is headline:
1. **Find drift** — Triage answers "anything to look at today?" in <10s with a **low
   false-positive rate** (James' #1 complaint: "it flags everything").
2. **Show evidence** — Model page delivers, without further digging, the baseline-vs-recent
   snapshot for the **three evidence metrics James hands engineers** — *untrimmed resistance,
   linearity error, electrical angle* — plus the metric that triggered the alert, and makes
   them **copy/paste-able and exportable**.
3. **Feed the system** — Process ingests trim / Final Test / Smoothness files.

Supporting, kept, not headline: per-unit pre/post-trim chart drill-down, Excel export,
Smoothness (now a Model tab), per-model specs/exclude_points (now in Settings), the
per-unit failure predictor (demoted to a collapsible diagnostic panel).

**Every task in every sub-plan must be traceable to one of those.** If a widget doesn't
help find drift, show evidence, or feed the system, it doesn't ship in v1.

---

## 1. Verified API reference (trust this, not the original plans)

### 1.1 Drift public API — `src/laser_trim_analyzer/ml/manager.py`

All return materialized dataclasses/dicts — **ORM-safe after the session closes.**

```python
get_drifting_models(db, sensitivity_preset: str = "standard") -> List[ModelAlertSummary]
    # ml/manager.py:1362. Reads model_metric_state, runs worst-of aggregation,
    # returns ONLY models with tier > STABLE, sorted (tier desc, magnitude desc).
    # ⚠ sensitivity_preset is IGNORED (preserved for API symmetry). Triage always
    #   reflects the CURRENTLY-TRAINED thresholds. To change what Triage shows you
    #   must retrain or call apply_sensitivity_preset (see §4.2).

get_model_drift_status(db, model: str) -> ModelDriftStatus
    # ml/manager.py:1408. Opens its OWN session, materializes inside it (ORM-safe).
    # .per_metric is Dict[str, MetricStatus] with all 8 WATCHED_METRICS.

preview_alert_count(db, sensitivity_preset: str) -> dict
    # ml/manager.py:1491. Returns {"warning": int, "drift": int, "out_of_control": int}.
    # Cheap what-if: recomputes thresholds from the candidate preset against CACHED
    # runtime state. Use for the Settings live preview. ORM-safe.
```

### 1.2 Drift training — `src/laser_trim_analyzer/ml/drift_training.py`

```python
train_drift_detector(
    db,
    sensitivity_preset: str = "standard",
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> TrainingSummary
    # Long-running / blocking. Re-scans history, recomputes baselines + thresholds,
    # upserts model_metric_state. progress_callback(model_name, models_done, models_total).
    # MIN_BASELINE_SAMPLES = 30 (below that a (model,metric) row is written is_trained=False).

_TRACK_METRIC_COLUMNS: dict[str, Column]   # module-level, drift_training.py:40
    # metric name -> TrackResult column. The UI MUST reuse this for the focus chart
    # (see §3 QA rule Q4). linearity_error -> TrackResult.final_linearity_error_shifted.
    # 'max_smoothness_value' is NOT in this dict — it lives on SmoothnessResult.
    # 3c adds a public alias TRACK_METRIC_COLUMNS = _TRACK_METRIC_COLUMNS (see §4.3).
```

### 1.3 Types — `src/laser_trim_analyzer/ml/drift_types.py`

```python
class DriftTier(IntEnum):  STABLE=0, WARNING=1, DRIFT=2, OUT_OF_CONTROL=3   # comparable with <,>
class AlertType(Enum):     STEP_CHANGE="step_change", SLOW_DRIFT="slow_drift"

WATCHED_METRICS: Tuple[str,...] = (   # exactly 8 — drives the 8-pill glance
    "sigma_gradient", "untrimmed_sigma_gradient", "untrimmed_resistance",
    "linearity_error", "measured_electrical_angle", "trim_pass_count",
    "resistance_change_percent", "max_smoothness_value",
)

@dataclass MetricStatus:        metric, tier, alert_type|None, magnitude, baseline_mean,
                                baseline_std, recent_mean|None, recent_count, is_trained
@dataclass ModelDriftStatus:    model, overall_tier, worst_metric|None, worst_alert_type|None,
                                per_metric: Dict[str,MetricStatus], last_processed|None
@dataclass ModelAlertSummary:   model, tier, alert_type, worst_metric, magnitude
@dataclass TrainingSummary:     models_trained, metrics_per_model=8,
                                skipped_insufficient_data: List[Tuple], duration_seconds

target_fp_for_tier(preset: str, tier: DriftTier) -> float    # drift_types.py:41
_PRESET_FP_MATRIX: presets = ["loose","standard","tight","strict"]   # drift_types.py:33
```

`compute_thresholds(sigma, target_fp) -> (h, L, z)` lives in
`src/laser_trim_analyzer/ml/multi_metric_drift_detector.py:42` (NOT drift_types).

### 1.4 model_metric_state ORM — `src/laser_trim_analyzer/database/models.py:1154`

Class `ModelMetricState`, unique `(model, metric)`. Columns:
`id, model, metric, baseline_cutoff_date, baseline_mean, baseline_std, baseline_count,
is_trained, h_warning, L_warning, z_warning, h_drift, L_drift, z_drift, h_oc, L_oc, z_oc,
ewma_state, cusum_pos, cusum_neg, last_updated`.

### 1.5 Core processing — `src/laser_trim_analyzer/core/`

```python
Processor(config: Optional[Config] = None, use_ml: bool = True)   # processor.py:68
    # ⚠ NO db parameter. It calls get_database() internally for ML thresholds and
    #   for saving Final-Test / Smoothness results.

Processor.process_batch(
    file_paths: List[Path],
    progress_callback: Optional[Callable[[ProcessingStatus], None]] = None,
    incremental: bool = True,
) -> Generator[AnalysisResult, None, BatchSummary]
    # Generator: YIELDS one AnalysisResult per processed (non-skipped) file;
    # RETURNS BatchSummary via StopIteration.value. Skipped files do NOT yield —
    # they are reported only via the progress_callback (status="skipped") and counted
    # in BatchSummary.skipped.

class ProcessingStatus(BaseAnalysisModel):   # core/models.py:325
    filename: str
    status: str            # "pending"|"processing"|"completed"|"failed"|"skipped"
    message: Optional[str] = None
    progress_percent: float = 0.0      # ⚠ use THIS for the progress bar
    result: Optional[AnalysisResult] = None   # set when status=="completed"
    error: Optional[str] = None
    # ⚠ NO current_file_index, NO total_files.

class AnalysisStatus(str, Enum):   # core/models.py:28
    PASS="Pass", FAIL="Fail", WARNING="Warning", ERROR="Error", UNTRIMMED="Untrimmed"
    # ⚠ NO SKIPPED member. Skips are a ProcessingStatus.status string, never a result status.

class BatchSummary(BaseAnalysisModel):   # core/models.py:341
    total_files, processed, passed, failed, warnings, skipped, errors, anomalies,
    start_time, end_time, total_processing_time, avg_sigma_gradient, pass_rate, high_risk_count

AnalysisResult.metadata.{filename, model, serial, file_date}  # metadata is FileMetadata
AnalysisResult.overall_status: AnalysisStatus
AnalysisResult.file_type: str   # "trim" | "final_test" | "smoothness"
```

**Persisting a processed file (Process page):**
```python
db.save_analysis(result) -> int   # manager.py:1080. GUI calls this ONLY for trim results.
# Final-Test and Smoothness results are already saved inside process_batch; do NOT
# re-save them. Guard: if result.file_type == "trim": db.save_analysis(result)
```

### 1.6 TrackResult ORM columns used by the UI — `database/models.py:~325`

`analysis_id` (FK → analysis_results.id), `sigma_gradient`, `untrimmed_sigma_gradient`,
`untrimmed_resistance`, `resistance_change_percent`, `measured_electrical_angle`,
`final_linearity_error_shifted`, `trim_pass_count`, and the SafeJSON arrays for the
per-unit chart: `position_data`, `error_data`, `upper_limits`, `lower_limits`,
`untrimmed_positions`, `untrimmed_errors`.

### 1.7 SmoothnessResult ORM — `database/models.py:1264`

`filename, file_path, file_hash, file_date, model, serial, element_label, test_date,
overall_status, smoothness_spec, max_smoothness_value, avg_smoothness_value,
smoothness_pass, timestamp, processing_time, linked_trim_id, match_confidence, ...`

### 1.8 AnalysisResult ORM — `database/models.py:188` (for test fixtures)

Required-ish columns when hand-constructing rows in tests:
`filename(NN), file_path, file_hash, file_date, model(NN), serial(NN),
system(Enum SystemType, NN), has_multi_tracks(NN), overall_status(Enum StatusType, NN),
processing_time, timestamp(NN default utc_now)`.

```python
class SystemType(PyEnum): A="A", B="B"                       # models.py:131
class StatusType(PyEnum): PASS="Pass", FAIL, WARNING, ERROR, PROCESSING_FAILED, UNTRIMMED  # models.py:137
# ⚠ Persisted as the enum NAME ('PASS','UNTRIMMED'). Server-side comparisons must use
#   StatusType.X.name or the bare string, never .value. (For display, .value "Pass" is fine.)
```

### 1.9 Config & singletons

```python
get_config() -> Config            # config.py:259, singleton
get_database() -> DatabaseManager # database/__init__ → manager.py, singleton on get_config().database.path
Config is a @dataclass (NOT Pydantic).
  config.database.path, .ensure_directory()
  config.processing.turbo_mode_threshold (=100), .incremental, .generate_plots
  config.ml.drift_sensitivity ("standard"; one of loose/standard/tight/strict),
            .enabled, .use_threshold_optimizer, .use_drift_detector, .min_samples_for_training
  config.gui.window_width(1400), .window_height(900), .theme
  config.active_models.{mps_models, model_prices, cost_ratio, recent_days, ...}
Config.save(config_path=None) / Config.load(config_path=None)   # config.py:191 / 135
```

```python
from laser_trim_analyzer.utils.threads import get_thread_manager   # threads.py:169
get_thread_manager().start_thread(target=fn, name="...")           # optional convenience
```

Per-model ML (Settings → ML Training; verified in `ml/manager.py`):
```python
from laser_trim_analyzer.ml import MLManager, TrainingProgress
mgr = MLManager(db)
results = mgr.train_all_models(min_samples=20,
              progress_callback=lambda p: ...)   # TrainingProgress: .models_complete/.models_total/.message
mgr.save_all()                                   # → Dict[str, ModelTrainingResult]
mgr.predictors: Dict[str, ModelPredictor]        # populated after get_shared_ml_manager(db).load_all()
get_shared_ml_manager(db, max_age_seconds=300.0) -> MLManager   # cached, load_all() already called (read path)
```

---

## 2. Corrected cross-cutting contracts (all sub-specs use these verbatim)

### 2.1 `PageBase` — pass `app`, build header actions with the right parent

The original PageBase sketch (`__init__(self, master, theme)`) gave pages no access to
db/config/navigation, and its `_PageHeader` tried to **reparent** action widgets via
`widget.master = actions_frame` — which **does not work in Tk** (the parent is fixed at
construction; no code in this repo reparents). Corrected contract (Spec 3a builds this):

```python
class PageBase(ctk.CTkFrame):
    page_title: str = "Untitled"

    def __init__(self, master, *, theme: "ThemeManager", app=None,
                 page_title: Optional[str] = None, **kwargs):
        super().__init__(master, fg_color=theme.SURFACE, corner_radius=0, **kwargs)
        self.theme = theme
        self.app = app                      # V6App | None (None OK for pure-widget tests)
        if page_title is not None:
            self.page_title = page_title
        self._build_chrome()                # builds header + an empty actions frame,
                                            #   then calls self.header_actions(actions_frame)
        self.build_content(self._content)

    def build_content(self, parent): raise NotImplementedError
    def header_actions(self, parent) -> None:        # OPTIONAL override.
        """Construct action widgets WITH `parent` as their master and pack them
        into `parent` (right-aligned). Default: no actions."""
        return None
    def on_show(self): pass
    def on_hide(self): pass

    # Shared thread-safety helper (see §2.4). Every page/section uses this instead
    # of a bare self.after for cross-thread UI updates.
    def safe_after(self, fn, delay=0):
        try:
            if self.winfo_exists():
                self.after(delay, lambda: fn() if self.winfo_exists() else None)
        except Exception:
            pass
```

`_PageHeader` builds: title label (left) + a transparent `actions_frame` packed right.
PageBase calls `self.header_actions(self._header.actions_frame)`; subclasses construct
their buttons/menus **with that frame as master** and `pack(side="right")` them. No
reparenting anywhere.

Pages stash app-derived handles in `build_content`/`__init__` as needed:
`self.app.db`, `self.app.config`, `self.app.show_page`, `self.app.set_model_route`, etc.

### 2.2 `V6App` — shared DB, dark-mode in `__init__`, test/auto-train opt-out

```python
class V6App(ctk.CTk):
    def __init__(self, config, db=None, auto_train_on_first_run=True):
        super().__init__()
        # Appearance is set HERE, not at module import (the original module-level
        # ctk.set_appearance_mode polluted V5 and test runs).
        ctk.set_appearance_mode("dark"); ctk.set_default_color_theme("blue")
        self.config = config
        self.theme = ThemeManager()
        # Share ONE DatabaseManager with the rest of the app (Processor uses
        # get_database() internally). In production db is None -> get_database(),
        # so app.db IS the global singleton on config.database.path. Tests inject
        # an isolated DatabaseManager on a tmp_path.
        self.db = db if db is not None else get_database()
        self._model_route = None            # routing hint (model, focus_metric)
        self._auto_train_on_first_run = auto_train_on_first_run
        ... build sidebar + page container + 4 pages ...
        self.show_page("triage")
        if auto_train_on_first_run:
            self.after(500, self._maybe_run_first_startup_train)   # see §4.4 + decision D3

    def show_page(self, name): ...                 # container.show + sidebar.set_active; unknown no-op
    def set_model_route(self, model, focus_metric=None): self._model_route = (model, focus_metric)
    def consume_model_route(self) -> Optional[str]: ...        # pops model only (3b)
    def consume_model_route_full(self) -> Tuple[Optional[str], Optional[str]]: ...  # pops (model, focus) (3c)
```

### 2.3 Theme additions (Spec 3a `theme.py`)

Keep all tokens from the spec3 design (`BG, SURFACE, CARD, ...`, tier colors, fonts,
spacing, radii). **Add three correctness helpers:**

```python
def tier_color(self, tier) -> (bg, fg):          # as designed; STABLE -> (SURFACE, TEXT_PRIMARY)
def tier_dot_color(self, tier) -> str:
    # FIX for the invisible-dot bug: STABLE on a SURFACE row renders nothing because
    # TIER_STABLE == SURFACE. Browse-list dots use THIS, which returns a muted-but-
    # visible color for STABLE (TEXT_DISABLED) and the tier accent otherwise.
    return self.TEXT_DISABLED if tier == DriftTier.STABLE else self.tier_color(tier)[1]
```

Typography: store the resolved family once. `FONT_FAMILY` stays a tuple, but pages read
`theme.font(size, weight)` which returns a `ctk.CTkFont` built from the first family that
the Tk font system reports available (falls back to `FONT_FAMILY[-1]`). This makes the
"Inter → Segoe UI → system-ui" fallback real instead of always using `[0]`.

Human-readable metric labels live in `drift_types` (see §4.3) so cards/pills/exports all
say "Untrimmed resistance", not `untrimmed_resistance`.

### 2.4 Threading rules (apply to every page/section/modal)

1. DB work runs on a background thread; **all widget mutation happens on the Tk thread**
   via `self.safe_after(...)` (winfo_exists-guarded — fixes the "after() on destroyed
   widget" TclError class).
2. **Stale-result guard (generation token).** Any control that can re-fire a reload while
   a previous reload is in flight (Model page: pill click, window change, model select;
   Triage: rapid re-show) increments `self._reload_gen` at kickoff, captures it in the
   worker, and the `safe_after` UI-apply bails if `gen != self._reload_gen`. This fixes
   the race where clicking pill B then A leaves A's slower query overwriting B.
3. Never block the Tk thread on a DB query > a few ms. Show a loading state if > 200ms.

### 2.5 Test infrastructure — one shared root, one app factory (`tests/conftest.py`)

The original plans duplicated a `tk_root` fixture in every file and created a second
`CTk()` *inside* `V6App` tests while a module root was live — two simultaneous CTk roots
is undefined behavior. Corrected:

```python
# tests/conftest.py
import sys, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

@pytest.fixture(scope="session")
def tk_root():
    import customtkinter as ctk
    try: ctk.deactivate_automatic_dpi_awareness()
    except Exception: pass
    root = ctk.CTk(); root.withdraw()
    yield root
    try: root.destroy()
    except Exception: pass

@pytest.fixture
def make_app(tmp_path):
    """Build a V6App on an ISOLATED tmp DB with auto-train OFF, destroyed on teardown.
    Use this for every V6App test — never construct V6App directly, and never combine
    an app test with the tk_root fixture in the same test (two roots)."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.gui.v6.app import V6App   # retirement (Spec 3e) changes only this line
    created = []
    def _factory(db_name="v6.db"):
        cfg = Config(); cfg.database.path = tmp_path / db_name
        app = V6App(cfg, db=DatabaseManager(cfg.database.path),
                    auto_train_on_first_run=False)
        app.withdraw(); created.append(app); return app
    yield _factory
    for app in created:
        try: app.destroy()
        except Exception: pass
```

Rules:
- **Widget tests** request `tk_root`, build the widget under it, assert, done.
- **App tests** request `make_app`, never `tk_root`. (pytest only instantiates a fixture
  when requested; keeping them disjoint avoids two live roots.)
- No `mainloop()` in tests. `after()` callbacks do not fire without a running loop — so
  any test of background-thread → `safe_after` paths must call the page's synchronous
  reload helper directly (e.g. `page.reload_now()`), not rely on the thread.

Import path during development is `laser_trim_analyzer.gui.v6.*`. The conftest `make_app`
references it through a thin indirection so the retirement move (gui/v6 → gui) is a
one-line conftest change, not a sweep of every test.

---

## 3. QA / data-analysis / manufacturing correctness rules (binding on all pages)

These encode AS9100 / SPC discipline and this product's domain facts. Each has a test.

- **Q1 — Zero-tolerance linearity is sacrosanct.** The per-unit chart shows **every**
  measurement point and **all** fail points; never smooth, downsample, or round in a way
  that could hide an out-of-spec point. The Units modal plots raw `position_data`/`error_data`
  with the stored `upper_limits`/`lower_limits` and marks fail points (reuse V5's
  `plot_error_vs_position(..., fail_points=...)`).
- **Q2 — A repeated serial is valid data, not a duplicate.** A unit can be trimmed
  multiple times. The Units tab and any unit list show **all** rows; never dedupe by serial.
- **Q3 — UNTRIMMED / NULL handling.** Test-sweep records have `sigma_gradient = NULL`
  (no post-trim) but a populated `untrimmed_sigma_gradient`. Every metric series query
  filters NULL for that metric (`col.isnot(None)`); never average or baseline across NULL.
  Spec 2 already does this server-side — the UI must not re-introduce NULLs into stats.
- **Q4 — Chart the signal the detector flagged.** The Model focus chart for a metric MUST
  plot the **same column** the drift baseline was trained on. Use `TRACK_METRIC_COLUMNS`
  (§4.3) — in particular `linearity_error → final_linearity_error_shifted` (TrackResult
  has no plain `linearity_error`; plotting the wrong column makes the baseline line not
  line up with the points). `max_smoothness_value → SmoothnessResult.max_smoothness_value`.
- **Q5 — Pair (date, value) before plotting.** Always build `[(d, v) for ... if d is not
  None and v is not None]` then unzip. Filtering dates and values with independent
  comprehensions can misalign the series (off-by-one against a NULL).
- **Q6 — σ-unit honesty.** A card/pill magnitude is `ModelAlertSummary.magnitude` /
  `MetricStatus.magnitude` = "σ-units beyond the tier's threshold." Render it as
  `+X.Xσ` with the tier name (e.g. "+4.2σ over Drift limit"), not as a bare distance from
  baseline. Don't imply a different statistic than the detector computed.
- **Q7 — SPC visual vocabulary (Rule 1 only).** The focus chart overlays: baseline mean
  (dashed), a ±2σ band (light, "warning zone"), ±3σ lines ("control limits"), and a recent-
  window highlight; points beyond ±3σ are drawn in the OOC color. Western Electric Rule 1
  only (anchor/redesign non-goal bars rules 2–8). Tier colors are exactly V5's semantic
  (Stable/Warning/Drift/Out-of-Control) so history-readers don't re-orient.
- **Q8 — Evidence is traceable.** "Copy summary" and "Export evidence pack" include model,
  the per-metric baseline mean ± std, recent mean, Δσ, alert type, time window, and (for
  the Excel pack) the flagged-batch unit list with serials + dates + raw values. This is
  what goes to engineers/MEs and must be auditable.
- **Q9 — No fabricated state.** Every page has an explicit empty state (Triage: "All
  models within tolerance — last processed {date}" / "Process files to get started";
  Model with no selection: "Pick a model"; Model with no data: "No measurements for this
  model yet"). Never render 0 / "—" placeholders as if they were measured values.
- **Q10 — Don't silently drop coverage.** Lists capped for performance (e.g. Units/browse
  at 200 rows) must say so ("showing 200 of N — narrow with search"), never imply the full
  set is shown.

---

## 4. New / changed data-layer helpers (small, well-specified)

### 4.1 `list_known_models(db) -> List[ModelSummary]` (Spec 3b) — **single-query, no N+1**

The original did `for model in all_models: get_model_drift_status(db, model)` — one session
per model (50+ sessions on every Triage show). Corrected design:

```python
@dataclass  # add to drift_types.py
class ModelSummary:
    model: str
    tier: DriftTier
    last_processed: Optional[datetime] = None

def list_known_models(db) -> List[ModelSummary]:
    # 1 session: distinct model + MAX(file_date) from analysis_results, UNION distinct
    #   model + MAX(file_date) from smoothness_results, merged in Python (keep latest date).
    # Tiers come from a SINGLE get_drifting_models(db) call (flagged models only);
    #   every other known model defaults to DriftTier.STABLE.
    # Returns ModelSummary list sorted by model. Total cost: 1 inventory session +
    #   1 get_drifting_models call — independent of model count.
```

### 4.2 `apply_sensitivity_preset(db, preset) -> int` (Spec 3d)

Because `get_drifting_models` ignores its preset arg, "Save preset" must persist new
thresholds so Triage actually changes. Recompute h/L/z in place from each trained row's
cached `baseline_std`; **preserve** baseline_*, cusum_pos/neg, ewma_state (no history
re-scan). Imports are correct as written in the original 3d plan:
`from laser_trim_analyzer.ml.multi_metric_drift_detector import compute_thresholds` and
`from laser_trim_analyzer.ml.drift_types import target_fp_for_tier, DriftTier`. Skips
`is_trained == False` / `baseline_std is None` rows. Returns rows updated.

### 4.3 Shared metric maps/labels (Spec 3c adds these; reused everywhere)

In `drift_training.py` add a public alias so the UI can't drift from the detector:
```python
TRACK_METRIC_COLUMNS = _TRACK_METRIC_COLUMNS   # public re-export
```
In `drift_types.py` add human labels (used by cards, pills, the drift table, exports):
```python
METRIC_LABELS = {
    "sigma_gradient": "Sigma gradient (post-trim)",
    "untrimmed_sigma_gradient": "Sigma gradient (untrimmed)",
    "untrimmed_resistance": "Untrimmed resistance",
    "linearity_error": "Linearity error",
    "measured_electrical_angle": "Electrical angle",
    "trim_pass_count": "Trim pass count",
    "resistance_change_percent": "Resistance change %",
    "max_smoothness_value": "Smoothness (max)",
}
def metric_label(metric: str) -> str: return METRIC_LABELS.get(metric, metric)
```

---

## 5. Decision log (structural changes vs the original plans)

- **D1 — V5 retirement is deferred and decoupled from Spec 3e.** The anchor's rule is
  "demotion before deletion … James worked hard to get the app where it is," and the
  redesign says V6 graduates "when it demonstrably serves the mission better." Deleting 13
  files + moving 20 + rewriting imports in one atomic commit, before parity is proven, is
  both high-risk and premature. **Spec 3e rewrites the Process page and hardens parity;
  the destructive promotion/deletion becomes a separate, gated, multi-commit "Graduation"
  procedure (Spec 3e §"Graduation") that James triggers explicitly after using V6.** Until
  then `--v6` selects V6 and `main`/no-flag stays V5.
- **D2 — Settings sections are really ported, not stubbed.** The original 3d left Per-model
  Specs / Pricing / Database Cleanup as placeholders pointing at the V5 page — which become
  dead links the moment V5 is deleted (D1 mitigates the timing, but parity still requires
  them). The mission keeps these as admin features; exclude_points specifically feeds ML
  correctness (CLAUDE.md). 3d ports all five sections for real, reusing existing DB/config
  methods (mostly UI wiring). ML Training ports drift **and** the existing threshold-
  optimizer / predictor / profiler controls so per-model ML isn't lost.
- **D3 — First-startup auto-train is opt-out and data-gated.** `V6App(..., auto_train_on_first_run=True)`;
  tests pass `False`. It runs only when `model_metric_state` is empty **and** there's data
  to train on (`analysis_results` non-empty). The decision is a pure, testable method;
  the modal is injectable (a `train_fn`) so widget tests never spawn real training threads.
- **D4 — Mission payoff features ship in 3c, not "a future spec."** Model page gets: a
  working **model selector** (James investigates arbitrary models on tips — success
  criterion #3), a working **"Copy summary"** (the daily paste-to-engineers action), the
  **per-unit pre/post-trim chart modal** (an explicitly-kept feature), and a basic
  **evidence-pack Excel** export. A permanently-empty predictor panel is replaced by a
  lazily-loaded, clearly-labeled diagnostic that degrades gracefully.
- **D5 — One shared metric→column map and label map.** Eliminates the silent
  wrong-column bug (Q4) and the raw-key UI strings.

---

## 6. Per-plan defect → fix index

| ID | Sub-spec | Defect | Fix (where) |
|----|----------|--------|-------------|
| C1 | 3e | `AnalysisStatus.SKIPPED` doesn't exist | drive skipped/progress from `progress_callback`/`BatchSummary` (§1.5) |
| C2 | 3e | `ProcessingStatus.current_file_index` doesn't exist | use `progress_percent` + `filename` (§1.5) |
| — | 3e | **trim results never saved** (functional regression) | `if file_type=="trim": db.save_analysis(result)` (§1.5) |
| I7 | 3e | `Processor(db=...)` — no such param | `Processor(config=app.config)` (§1.5) |
| C3 | 3b | `list_known_models` N+1 | single-query helper (§4.1) |
| C4 | 3d | auto-train fires in tests; its own test can't pass | opt-out flag + data gate + pure decision (§2.2, D3) |
| C5/C6 | 3a | widget reparenting `widget.master=` | `header_actions(parent)` (§2.1) |
| C7 | all | two live CTk roots in tests | shared `tk_root` + `make_app` (§2.5) |
| I1 | 3d/3e | Settings placeholders → dead links after deletion | real ports + deferred retirement (D1, D2) |
| I2 | 3c | `_window_var`/`_window_choice` drift | single source via the segmented control |
| I3 | 3c | reload thread races | generation token (§2.4) |
| I4 | 3a/3b | invisible STABLE tier dot | `tier_dot_color` (§2.3) |
| I8 | 3c | DetachedInstanceError in `_load_smoothness` | materialize to dicts in-session (Q-pattern) |
| I9 | all | `after()` on destroyed widget | `safe_after` (§2.1/2.4) |
| I10 | 3e | retirement commit too large/atomic | decomposed, gated Graduation (D1) |
| Q4 | 3c | focus chart plots wrong column for linearity | `TRACK_METRIC_COLUMNS` (§3 Q4, §4.3) |
| Q5 | 3c | date/value misalignment | pair-then-unzip (§3 Q5) |
| S6 | 3c | no Model empty state / **no model selector** | empty state + selector (D4) |
| S7 | 3c | permanently-empty predictor panel | lazy, labeled, graceful (D4) |
| M5 | 3a | module-level `set_appearance_mode` | move into `V6App.__init__` (§2.2) |
| M3 | 3a | font fallback never used (always `[0]`) | `theme.font()` resolves availability (§2.3) |
| — | 3c | per-unit modal + Copy summary + evidence export deferred | shipped in 3c (D4) |

---

## 7. Execution order & regression gate

Sequence unchanged: **3a → 3b → 3c → 3d → 3e** (Process rewrite) → **Graduation (gated)**.
After each sub-spec, the running regression sweep must stay 0-fail:

```
pytest tests/test_spec1_untrimmed_sigma.py \
       tests/test_log_derived_bugfixes_2026_05_30.py \
       tests/test_5_8_2026_bugfixes.py \
       tests/test_spec2_multi_metric_drift.py \
       tests/test_spec3a_shell.py tests/test_spec3b_triage.py \
       tests/test_spec3c_model.py tests/test_spec3d_settings.py \
       tests/test_spec3e_process.py   # add files as sub-specs land
```

**Graduation gate (all must be true before any V5 deletion):**
1. All five sub-spec test files green + full `pytest tests/` green.
2. Manual V6 smoke: every page renders; Triage empty + populated states; Model selector
   reaches an arbitrary model; per-unit modal opens; Copy summary + evidence Excel work;
   Settings all five sections functional (preset save changes Triage; specs/pricing/cleanup
   act on the DB; ML retrain runs).
3. James has run V6 against a real DB copy and confirmed it answers the mission.
Only then run the decomposed promotion (Spec 3e §Graduation).
</content>
</invoke>
