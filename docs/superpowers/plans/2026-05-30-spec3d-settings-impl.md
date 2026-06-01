# Spec 3d — Settings Restructure Implementation Plan (rewritten 2026-06-01)

> **READ FIRST:** `docs/superpowers/plans/2026-06-01-spec3-rewrite-foundations.md`.
> Implements decision **D2** (port the four admin sections for real — Specs/Pricing/Cleanup/ML — so V5
> deletion never strands a feature) and **D3** (auto-train is opt-out + data-gated + testable). The
> primary "fix flags everything" surface is **Alert Thresholds** (sensitivity preset + live preview +
> Save that actually changes Triage via `apply_sensitivity_preset`). Shared fixtures in `tests/conftest.py`.

**Goal:** Replace the Settings placeholder with a scrollable list of 5 collapsible sections —
**Alert Thresholds** (expanded), **Per-model Specs**, **ML Training**, **Pricing**, **Database Cleanup** —
each really functional. Wire the data-gated first-startup auto-train.

**Target branch:** `V6`. Start at the Spec 3c final commit.

**Fixes applied (foundations §6):** C4 (auto-train opt-out + data gate + pure decision; injectable
training so widget tests never spawn real threads), I1 (no dead "go use V5" placeholders — real ports),
I9 (`safe_after` guards in section threads), M8 (preset test asserts computed values, not "> 1.0").

**Verified V5 APIs reused (do not reinvent):**
`db.get_all_model_specs()`, `db.get_model_spec(model)`, `db.save_model_spec(data: dict) -> (id, created)`,
`db.delete_model_spec(model)`, `db.import_model_specs_from_excel(path) -> dict`,
`db.get_spec_discrepancies()`, `db.scan_database_health() -> dict`;
`core.analyzer.parse_exclude_points` / `format_exclude_points`;
`config.active_models.{model_prices, cost_ratio, recent_days, mps_models}` + `config.save()`;
`compute_thresholds` (ml/multi_metric_drift_detector), `target_fp_for_tier` (ml/drift_types).

---

## File Structure

**Created:** `gui/v6/widgets/{settings_card,sensitivity_slider,training_modal}.py`,
`gui/v6/sections/{__init__,alert_thresholds,per_model_specs,ml_training,pricing,database_cleanup}.py`,
`gui/v6/pages/settings_page.py`, `tests/test_spec3d_settings.py`.

**Modified:** `ml/manager.py` (`apply_sensitivity_preset`), `gui/v6/app.py` (data-gated auto-train hook +
register real SettingsPage).

---

## Task 1: `apply_sensitivity_preset(db, preset)` (foundations §4.2)

- [ ] **Step 1:** Create `tests/test_spec3d_settings.py`:

```python
"""Spec 3d — Settings. Foundations §4.2/D2/D3. Fixtures in tests/conftest.py."""
import pytest


def _seed_metric_state(db, model="T", baseline_std=0.001, trained=True):
    from datetime import datetime
    from laser_trim_analyzer.database.models import ModelMetricState
    with db.session() as s:
        s.add(ModelMetricState(
            model=model, metric="sigma_gradient", baseline_mean=0.01, baseline_std=baseline_std,
            baseline_count=100, is_trained=trained,
            h_warning=1.0, h_drift=5.0, h_oc=10.0, L_warning=1.0, L_drift=2.0, L_oc=3.0,
            z_warning=1.0, z_drift=2.0, z_oc=3.0, cusum_pos=2.5, cusum_neg=0.0, ewma_state=0.01,
            last_updated=datetime.now()))
        s.commit()


def test_apply_sensitivity_preset_recomputes_to_known_values(tmp_path):
    """M8 fix: assert the EXACT recomputed thresholds, not a vague '> 1.0'."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.ml.manager import apply_sensitivity_preset
    from laser_trim_analyzer.ml.multi_metric_drift_detector import compute_thresholds
    from laser_trim_analyzer.ml.drift_types import target_fp_for_tier, DriftTier

    db = DatabaseManager(tmp_path / "p.db")
    _seed_metric_state(db, baseline_std=0.001)
    n = apply_sensitivity_preset(db, "tight")
    assert n == 1
    exp_h, exp_L, exp_z = compute_thresholds(0.001, target_fp_for_tier("tight", DriftTier.WARNING))
    with db.session() as s:
        row = s.query(ModelMetricState).filter_by(model="T", metric="sigma_gradient").first()
        assert row.baseline_mean == pytest.approx(0.01)     # baseline untouched
        assert row.baseline_std == pytest.approx(0.001)
        assert row.cusum_pos == pytest.approx(2.5)          # runtime state preserved
        assert row.L_warning == pytest.approx(exp_L)
        assert row.z_warning == pytest.approx(exp_z)
        assert row.h_warning == pytest.approx(exp_h)


def test_apply_sensitivity_preset_skips_untrained(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.manager import apply_sensitivity_preset
    db = DatabaseManager(tmp_path / "u.db")
    _seed_metric_state(db, model="U", trained=False)
    assert apply_sensitivity_preset(db, "tight") == 0   # untrained skipped, no raise
```

- [ ] **Step 2:** Append to `ml/manager.py` (imports verified correct):

```python
def apply_sensitivity_preset(db, preset: str) -> int:
    """Recompute per-tier (h, L, z) in place from each trained row's cached baseline_std
    using `preset`'s target FP rates. Preserves baseline_* and runtime cusum/ewma state
    (no history re-scan). Returns rows updated. Lets Settings 'Save preset' change what
    Triage flags without a full retrain (get_drifting_models ignores its preset arg)."""
    from datetime import datetime
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.ml.drift_types import DriftTier, target_fp_for_tier
    from laser_trim_analyzer.ml.multi_metric_drift_detector import compute_thresholds

    updated = 0
    with db.session() as s:
        rows = s.query(ModelMetricState).filter(
            ModelMetricState.is_trained == True,                # noqa: E712
            ModelMetricState.baseline_std.isnot(None)).all()
        for row in rows:
            sigma = row.baseline_std
            for tier, (hc, lc, zc) in (
                (DriftTier.WARNING, ("h_warning", "L_warning", "z_warning")),
                (DriftTier.DRIFT, ("h_drift", "L_drift", "z_drift")),
                (DriftTier.OUT_OF_CONTROL, ("h_oc", "L_oc", "z_oc")),
            ):
                h, L, z = compute_thresholds(sigma, target_fp_for_tier(preset, tier))
                setattr(row, hc, h); setattr(row, lc, L); setattr(row, zc, z)
            row.last_updated = datetime.now()
            updated += 1
        s.commit()
    return updated
```

- [ ] **Step 3:** Run `-k apply_sensitivity` → PASS. Commit `feat(spec3d): apply_sensitivity_preset
  (in-place threshold recompute, runtime state preserved)`.

---

## Task 2: SettingsCard

- [ ] **Step 1:** Append tests (collapsed default; `expanded=True` starts open; `toggle`; `body_frame`):

```python
# ---- Task 2: SettingsCard -------------------------------------------------

def test_settings_card_states(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.settings_card import SettingsCard
    t = ThemeManager()
    assert SettingsCard(tk_root, theme=t, title="A")._expanded is False
    c = SettingsCard(tk_root, theme=t, title="B", expanded=True)
    assert c._expanded is True
    c.toggle(); assert c._expanded is False


def test_settings_card_body_frame_is_fillable(tk_root):
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.settings_card import SettingsCard
    c = SettingsCard(tk_root, theme=ThemeManager(), title="C", expanded=True)
    ctk.CTkLabel(c.body_frame(), text="hi").pack()   # no raise
```

- [ ] **Step 2:** Create `widgets/settings_card.py` (same as original draft; fonts via `theme.font`):

```python
"""Spec 3d — SettingsCard: collapsible section host."""
import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


class SettingsCard(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, title: str, expanded: bool = False, **kwargs):
        super().__init__(master, fg_color=theme.CARD, corner_radius=theme.RADIUS_MD, **kwargs)
        self.theme = theme; self._expanded = False
        header = ctk.CTkFrame(self, fg_color="transparent"); header.pack(side="top", fill="x")
        self._title = ctk.CTkLabel(header, text=title, font=theme.font(theme.SIZE_HEADING, "bold"),
                                   text_color=theme.TEXT_PRIMARY, anchor="w")
        self._title.pack(side="left", fill="x", expand=True, padx=theme.SPACE_MD, pady=theme.SPACE_SM)
        self._chevron = ctk.CTkButton(header, text="▾", width=32, fg_color="transparent",
                                      hover_color=theme.ELEVATED, text_color=theme.TEXT_SECONDARY,
                                      font=theme.font(theme.SIZE_HEADING), command=self.toggle,
                                      corner_radius=theme.RADIUS_SM)
        self._chevron.pack(side="right", padx=theme.SPACE_MD, pady=theme.SPACE_XS)
        for w in (header, self._title):
            w.bind("<Button-1>", lambda e: self.toggle())
        self._body = ctk.CTkFrame(self, fg_color="transparent")
        if expanded:
            self.toggle()

    def body_frame(self) -> ctk.CTkFrame:
        return self._body

    def toggle(self) -> None:
        if self._expanded:
            self._body.pack_forget(); self._chevron.configure(text="▾")
        else:
            self._body.pack(side="top", fill="x", padx=self.theme.SPACE_MD,
                            pady=(0, self.theme.SPACE_SM))
            self._chevron.configure(text="▴")
        self._expanded = not self._expanded
```

- [ ] **Step 3:** Run `-k settings_card` → PASS. Commit `feat(spec3d): SettingsCard`.

---

## Task 3: SensitivitySlider

- [ ] **Step 1:** Append:

```python
# ---- Task 3: SensitivitySlider --------------------------------------------

def test_sensitivity_slider(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.sensitivity_slider import SensitivitySlider
    got = []
    s = SensitivitySlider(tk_root, theme=ThemeManager(), initial="standard", on_change=got.append)
    assert s.value() == "standard"
    s.set_value("tight"); assert s.value() == "tight"
```

- [ ] **Step 2:** Create `widgets/sensitivity_slider.py` (CTkSegmentedButton, 4 stops):

```python
"""Spec 3d — SensitivitySlider: 4-stop preset picker (loose/standard/tight/strict)."""
from typing import Callable

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager

_PRESETS = ["loose", "standard", "tight", "strict"]


class SensitivitySlider(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, initial: str, on_change: Callable[[str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme; self._on_change = on_change
        self._seg = ctk.CTkSegmentedButton(self, values=_PRESETS, command=on_change,
                                           fg_color=theme.CARD, selected_color=theme.ACCENT,
                                           selected_hover_color=theme.ACCENT_HOVER,
                                           unselected_color=theme.CARD, unselected_hover_color=theme.ELEVATED,
                                           text_color=theme.TEXT_PRIMARY, corner_radius=theme.RADIUS_SM)
        self._seg.pack(side="top", fill="x")
        self._seg.set(initial if initial in _PRESETS else "standard")

    def value(self) -> str:
        return self._seg.get()

    def set_value(self, preset: str) -> None:
        if preset in _PRESETS:
            self._seg.set(preset)
```

- [ ] **Step 3:** Run `-k sensitivity_slider` → PASS. Commit `feat(spec3d): SensitivitySlider`.

---

## Task 4: TrainingModal (injectable training so tests stay deterministic)

- [ ] **Step 1:** Append:

```python
# ---- Task 4: TrainingModal ------------------------------------------------

def test_training_modal_uses_injected_train_fn(tk_root, tmp_path):
    """Inject a fake train_fn so the widget test never runs real training threads."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.training_modal import TrainingModal
    db = DatabaseManager(tmp_path / "tm.db")
    called = {}
    def fake_train(d, preset, progress_callback=None):
        called["preset"] = preset
        if progress_callback:
            progress_callback("M1", 0, 1)
        class _S: pass
        return _S()
    modal = TrainingModal(tk_root, theme=ThemeManager(), db=db, preset="standard", train_fn=fake_train)
    modal._run_training()           # synchronous path (no thread) for the test
    assert called["preset"] == "standard"
    modal.destroy()
```

- [ ] **Step 2:** Create `widgets/training_modal.py` (default `train_fn=train_drift_detector`;
  `start()` spawns the thread, `_run_training()` is callable directly):

```python
"""Spec 3d — TrainingModal: drift training with progress. train_fn injectable for tests."""
import threading
from typing import Callable, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


class TrainingModal(ctk.CTkToplevel):
    def __init__(self, master, theme: ThemeManager, db, preset: str,
                 train_fn: Optional[Callable] = None):
        super().__init__(master)
        self.theme = theme; self.db = db; self.preset = preset
        if train_fn is None:
            from laser_trim_analyzer.ml.drift_training import train_drift_detector
            train_fn = train_drift_detector
        self._train_fn = train_fn
        self.title("Training drift detector"); self.geometry("420x190")
        self.configure(fg_color=theme.SURFACE); self.transient(master)
        ctk.CTkLabel(self, text="Training drift detector", font=theme.font(theme.SIZE_HEADING, "bold"),
                     text_color=theme.TEXT_PRIMARY).pack(pady=(theme.SPACE_LG, theme.SPACE_SM))
        self._status = ctk.CTkLabel(self, text="Preparing…", font=theme.font(theme.SIZE_BODY),
                                    text_color=theme.TEXT_SECONDARY)
        self._status.pack(pady=theme.SPACE_SM)
        self._bar = ctk.CTkProgressBar(self, progress_color=theme.ACCENT, fg_color=theme.CARD)
        self._bar.pack(fill="x", padx=theme.SPACE_LG, pady=theme.SPACE_SM); self._bar.set(0)
        ctk.CTkButton(self, text="Close", fg_color=theme.CARD, hover_color=theme.ELEVATED,
                      text_color=theme.TEXT_PRIMARY, command=self.destroy,
                      corner_radius=theme.RADIUS_SM).pack(pady=theme.SPACE_MD)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def start(self) -> None:
        threading.Thread(target=self._run_training, daemon=True).start()

    def _run_training(self) -> None:
        try:
            self._train_fn(self.db, self.preset, progress_callback=self._on_progress)
        except Exception as exc:
            self._safe(lambda: self._status.configure(text=f"Training failed: {exc}")); return
        self._safe(lambda: self._status.configure(text="Training complete."))
        self._safe(lambda: self._bar.set(1.0))
        self._safe(lambda: self.after(750, self.destroy))

    def _on_progress(self, model: str, done: int, total: int) -> None:
        self._safe(lambda: self._status.configure(text=f"Training {done + 1} / {total} ({model})"))
        self._safe(lambda: self._bar.set((done + 1) / max(total, 1)))

    def _safe(self, fn) -> None:
        try:
            if self.winfo_exists():
                self.after(0, lambda: fn() if self.winfo_exists() else None)
        except Exception:
            pass
```

- [ ] **Step 3:** Run `-k training_modal` → PASS. Commit `feat(spec3d): TrainingModal (injectable train_fn)`.

---

## Task 5: Alert Thresholds section (the "fix flags everything" surface — full implementation)

- [ ] **Step 1:** Append a build test (use a tiny fake-app with `db`+`config`):

```python
# ---- Task 5: sections ------------------------------------------------------

class _FakeApp:
    def __init__(self, db, cfg):
        self.db = db; self.config = cfg

def _fake_app(tmp_path, name="x.db"):
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.database.manager import DatabaseManager
    cfg = Config(); cfg.database.path = tmp_path / name
    return _FakeApp(DatabaseManager(cfg.database.path), cfg)


def test_alert_thresholds_section_builds(tk_root, tmp_path):
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.sections.alert_thresholds import build_alert_thresholds_section
    parent = ctk.CTkFrame(tk_root)
    build_alert_thresholds_section(parent, theme=ThemeManager(), app=_fake_app(tmp_path))
```

- [ ] **Step 2:** Create `sections/__init__.py` (empty) and `sections/alert_thresholds.py`. Debounced
  live preview via `preview_alert_count`; Save persists `config.ml.drift_sensitivity` and calls
  `apply_sensitivity_preset` so **Triage actually changes**:

```python
"""Spec 3d — Alert Thresholds: sensitivity preset + live preview + Save. The primary tool
for reducing the false-positive rate ('flags everything')."""
import threading

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.sensitivity_slider import SensitivitySlider
from laser_trim_analyzer.ml.drift_types import WATCHED_METRICS, metric_label
from laser_trim_analyzer.ml.manager import apply_sensitivity_preset, preview_alert_count


def build_alert_thresholds_section(parent, theme: ThemeManager, app) -> None:
    t = theme
    current = getattr(app.config.ml, "drift_sensitivity", "standard")
    state = {"selected": current, "after_id": None}

    ctk.CTkLabel(parent, justify="left", wraplength=640, anchor="w", font=t.font(t.SIZE_BODY),
                 text_color=t.TEXT_SECONDARY,
                 text=("Tighter presets reduce false positives but may miss true drift; looser presets "
                       "surface more at the cost of noise. Preview shows how many models each preset "
                       "would flag against current data."))\
        .pack(side="top", fill="x", anchor="w", pady=(0, t.SPACE_MD))

    preview = ctk.CTkLabel(parent, text="Would flag: — Warning, — Drift, — Out-of-Control",
                           font=t.font(t.SIZE_BODY, "bold"), text_color=t.TEXT_PRIMARY, anchor="w")
    preview.pack(side="top", anchor="w", pady=(0, t.SPACE_SM))

    def refresh_preview(preset):
        def work():
            try:
                c = preview_alert_count(app.db, preset)
            except Exception:
                c = {"warning": 0, "drift": 0, "out_of_control": 0}
            txt = (f"Would flag at '{preset}': {c['warning']} Warning, "
                   f"{c['drift']} Drift, {c['out_of_control']} Out-of-Control")
            try:
                if preview.winfo_exists():
                    preview.after(0, lambda: preview.winfo_exists() and preview.configure(text=txt))
            except Exception:
                pass
        threading.Thread(target=work, daemon=True).start()

    def on_change(preset):
        state["selected"] = preset
        if state["after_id"] is not None:
            try: preview.after_cancel(state["after_id"])
            except Exception: pass
        state["after_id"] = preview.after(200, lambda: refresh_preview(preset))

    SensitivitySlider(parent, theme=t, initial=current, on_change=on_change)\
        .pack(side="top", fill="x", pady=(0, t.SPACE_MD))

    # Educational: what each watched metric means.
    desc = ctk.CTkFrame(parent, fg_color="transparent"); desc.pack(side="top", fill="x", pady=(0, t.SPACE_MD))
    for m in WATCHED_METRICS:
        ctk.CTkLabel(desc, text=f"• {metric_label(m)}", font=t.font(t.SIZE_CAPTION),
                     text_color=t.TEXT_SECONDARY, anchor="w").pack(side="top", fill="x")

    save_btn = ctk.CTkButton(parent, text="Save preset", fg_color=t.ACCENT, hover_color=t.ACCENT_HOVER,
                             text_color=t.TEXT_INVERSE, corner_radius=t.RADIUS_SM)

    def save():
        preset = state["selected"]
        app.config.ml.drift_sensitivity = preset
        try: app.config.save()
        except Exception: pass
        # Recompute thresholds so Triage reflects the new preset (get_drifting_models ignores its arg).
        threading.Thread(target=lambda: apply_sensitivity_preset(app.db, preset), daemon=True).start()
        save_btn.configure(text="Saved ✓")
        save_btn.after(1500, lambda: save_btn.winfo_exists() and save_btn.configure(text="Save preset"))

    save_btn.configure(command=save)
    save_btn.pack(side="top", anchor="w")
    refresh_preview(current)
```

- [ ] **Step 3:** Run `-k alert_thresholds` → PASS. Commit `feat(spec3d): Alert Thresholds section
  (preview + preset save that re-thresholds in place)`.

---

## Task 6: ML Training section (drift + existing per-model ML — D2)

Ports the V5 training entrypoints so per-model ML (threshold optimizer / predictor / profiler) is NOT
lost. Drift retrain uses TrainingModal. The other-models button reuses the V5 path.

- [ ] **Step 1:** Use the **verified** V5 training entrypoint (confirmed in `ml/manager.py`): construct a
  fresh `MLManager(db)` and call
  `train_all_models(min_samples=int, progress_callback=Optional[Callable[[TrainingProgress], None]])
  -> Dict[str, ModelTrainingResult]`, then `mgr.save_all()`. `TrainingProgress` exposes
  `.models_complete`, `.models_total`, `.message`. (This is exactly what V5 `settings._run_training`
  does.) Do NOT re-implement training logic — call this.

- [ ] **Step 2:** Append a build test, then create `sections/ml_training.py`:

```python
def test_ml_training_section_builds(tk_root, tmp_path):
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.sections.ml_training import build_ml_training_section
    build_ml_training_section(ctk.CTkFrame(tk_root), theme=ThemeManager(), app=_fake_app(tmp_path, "ml.db"))
```

```python
"""Spec 3d — ML Training: drift retrain (TrainingModal) + per-model ML retrain (V5 path)."""
import threading

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.training_modal import TrainingModal


def build_ml_training_section(parent, theme: ThemeManager, app) -> None:
    t = theme
    ctk.CTkLabel(parent, justify="left", wraplength=640, anchor="w", font=t.font(t.SIZE_BODY),
                 text_color=t.TEXT_SECONDARY,
                 text=("Retrain ML against current data. Drift baselines feed Triage/Model; the "
                       "per-model threshold optimizer, failure predictor, and profiler are the "
                       "existing per-model ML."))\
        .pack(side="top", fill="x", anchor="w", pady=(0, t.SPACE_MD))

    status = ctk.CTkLabel(parent, text="", font=t.font(t.SIZE_CAPTION),
                          text_color=t.TEXT_SECONDARY, anchor="w")
    status.pack(side="top", fill="x")

    def retrain_drift():
        preset = getattr(app.config.ml, "drift_sensitivity", "standard")
        TrainingModal(parent, theme=t, db=app.db, preset=preset).start()

    def retrain_per_model():
        # Verified V5 entrypoint: fresh MLManager(db).train_all_models(...) + save_all(). Off-thread.
        def work():
            try:
                from laser_trim_analyzer.ml import MLManager
                mgr = MLManager(app.db)
                results = mgr.train_all_models(
                    min_samples=getattr(app.config.ml, "min_samples_for_training", 20))
                mgr.save_all()
                msg = f"Per-model ML retrained: {len(results)} models."
            except Exception as exc:
                msg = f"Per-model ML retrain failed: {exc}"
            try:
                if status.winfo_exists():
                    status.after(0, lambda: status.winfo_exists() and status.configure(text=msg))
            except Exception:
                pass
        status.configure(text="Retraining per-model ML…")
        threading.Thread(target=work, daemon=True).start()

    ctk.CTkButton(parent, text="Retrain drift detector", command=retrain_drift, fg_color=t.ACCENT,
                  hover_color=t.ACCENT_HOVER, text_color=t.TEXT_INVERSE, corner_radius=t.RADIUS_SM)\
        .pack(side="top", anchor="w", pady=(0, t.SPACE_SM))
    ctk.CTkButton(parent, text="Retrain per-model ML (thresholds + predictor + profiler)",
                  command=retrain_per_model, fg_color=t.CARD, hover_color=t.ELEVATED,
                  text_color=t.TEXT_PRIMARY, corner_radius=t.RADIUS_SM)\
        .pack(side="top", anchor="w")
```

> **Implementer note:** `train_all_models` / `save_all` are the verified V5 calls. The build test only
> checks construction (training itself needs seeded data); add a smoke test against a small seeded DB if
> you want coverage of the real run.

- [ ] **Step 3:** Run + commit `feat(spec3d): ML Training section (drift + per-model ports)`.

---

## Task 7: Per-model Specs section (real port — exclude_points feeds ML correctness)

Reuses `db.get_all_model_specs / get_model_spec / save_model_spec / delete_model_spec` and
`parse_exclude_points / format_exclude_points`. exclude_points changes flip FAIL→PASS labels (CLAUDE.md),
so this is correctness-relevant, not cosmetic.

- [ ] **Step 1:** Read V5 `gui/pages/specs.py` for the field list and the exclude-points JSON↔display
  conversion (lines ~436–513). Port the same data contract.

- [ ] **Step 2:** Append a build test + a save round-trip test:

```python
def test_per_model_specs_section_builds(tk_root, tmp_path):
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.sections.per_model_specs import build_per_model_specs_section
    build_per_model_specs_section(ctk.CTkFrame(tk_root), theme=ThemeManager(), app=_fake_app(tmp_path, "sp.db"))
```

- [ ] **Step 3:** Create `sections/per_model_specs.py`: a model `CTkComboBox` (values from
  `db.get_all_model_specs()`), entries for `linearity_spec` and `exclude_points` (displayed via
  `format_exclude_points`, parsed back via `parse_exclude_points` and stored as JSON in the dict passed to
  `db.save_model_spec(data)`), Save and Delete buttons calling the real DB methods. Run all DB work off
  the Tk thread with `safe`-guarded callbacks. Mirror V5 specs.py field handling exactly.

- [ ] **Step 4:** Run + commit `feat(spec3d): Per-model Specs section (real port; exclude_points)`.

---

## Task 8: Pricing section (real port)

Reuses `config.active_models.{model_prices, cost_ratio, recent_days}` + `config.save()` and the V5
import-from-file logic.

- [ ] **Step 1:** Append a build test. Create `sections/pricing.py`: a cost-ratio entry (0.01–1.0), a
  recent-days entry, an "Import pricing from file" button (port V5 `_import_pricing` — flexible column
  match writing into `config.active_models.model_prices`), a "Clear pricing" button, and a count label.
  Save writes config + `config.save()`. Mirror V5 settings pricing subsection.

- [ ] **Step 2:** Run + commit `feat(spec3d): Pricing section (real port)`.

---

## Task 9: Database Cleanup section (real port)

Reuses `db.scan_database_health()` and the V5 cleanup delete paths.

- [ ] **Step 1:** Read V5 `gui/pages/settings.py` `_create_database_cleanup_section` / `_scan_database` /
  the "Clear Selected" handler to capture the exact delete methods (non-MPS / by-date / suspect-quality /
  reset-skipped). Append a build test. Create `sections/database_cleanup.py`: a "Scan database" button
  (calls `db.scan_database_health()` off-thread, shows counts), the same category checkboxes V5 has, and a
  "Clear selected" button calling the real delete methods off-thread with a confirm dialog. Mirror V5.

- [ ] **Step 2:** Run + commit `feat(spec3d): Database Cleanup section (real port)`.

---

## Task 10: SettingsPage + data-gated first-startup auto-train

- [ ] **Step 1:** Append:

```python
# ---- Task 10: SettingsPage + auto-train ------------------------------------

def test_settings_page_has_five_cards(make_app):
    app = make_app()
    page = app.page_container.get_page("settings")
    assert len(page._cards) == 5


def test_should_offer_first_startup_train_is_data_gated(make_app):
    """D3: empty DB → no offer (nothing to train). Data present + empty metric_state → offer."""
    from datetime import datetime
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR, SystemType, StatusType
    app = make_app()
    assert app._should_offer_first_startup_train() is False     # empty DB
    with app.db.session() as s:
        s.add(DBAR(filename="x.xls", file_path="/f/x.xls", file_hash="hx", model="M", serial="sn1",
                   system=SystemType.A, file_date=datetime.now(), timestamp=datetime.now(),
                   overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1))
        s.commit()
    assert app._should_offer_first_startup_train() is True      # data, no metric_state yet
```

- [ ] **Step 2:** Create `pages/settings_page.py`:

```python
"""Spec 3d — SettingsPage: scrollable list of 5 collapsible sections."""
import customtkinter as ctk

from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.sections.alert_thresholds import build_alert_thresholds_section
from laser_trim_analyzer.gui.v6.sections.database_cleanup import build_database_cleanup_section
from laser_trim_analyzer.gui.v6.sections.ml_training import build_ml_training_section
from laser_trim_analyzer.gui.v6.sections.per_model_specs import build_per_model_specs_section
from laser_trim_analyzer.gui.v6.sections.pricing import build_pricing_section
from laser_trim_analyzer.gui.v6.widgets.settings_card import SettingsCard


class SettingsPage(PageBase):
    page_title = "Settings"

    def __init__(self, master, *, theme, app, page_title="Settings"):
        self._cards = []
        super().__init__(master, theme=theme, app=app, page_title=page_title)

    def build_content(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        for title, expanded, build in (
            ("Alert Thresholds", True, build_alert_thresholds_section),
            ("Per-model Specs", False, build_per_model_specs_section),
            ("ML Training", False, build_ml_training_section),
            ("Pricing", False, build_pricing_section),
            ("Database Cleanup", False, build_database_cleanup_section),
        ):
            card = SettingsCard(scroll, theme=self.theme, title=title, expanded=expanded)
            card.pack(side="top", fill="x", pady=(0, self.theme.SPACE_SM))
            build(card.body_frame(), theme=self.theme, app=self.app)
            self._cards.append(card)
```

- [ ] **Step 3:** In `gui/v6/app.py`: add the data-gated decision + the hook, and register the real
  SettingsPage. The decision is pure/testable; the hook only shows the modal when warranted and the flag
  is on:

```python
    def _should_offer_first_startup_train(self) -> bool:
        """True only when there is data to train on AND model_metric_state is empty."""
        from laser_trim_analyzer.database.models import AnalysisResult as DBAR, ModelMetricState
        try:
            with self.db.session() as s:
                has_data = s.query(DBAR.id).first() is not None
                trained = s.query(ModelMetricState.id).first() is not None
            return has_data and not trained
        except Exception:
            return False

    def _maybe_run_first_startup_train(self) -> None:
        if not self._auto_train_on_first_run:
            return
        if not self._should_offer_first_startup_train():
            return
        from laser_trim_analyzer.gui.v6.widgets.training_modal import TrainingModal
        preset = getattr(self.config.ml, "drift_sensitivity", "standard")
        TrainingModal(self, theme=self.theme, db=self.db, preset=preset).start()
```

In `_build_pages()` register the real SettingsPage (replace its placeholder):

```python
        from laser_trim_analyzer.gui.v6.pages.settings_page import SettingsPage
        self.page_container.add_page(
            "settings", SettingsPage(self.page_container, theme=self.theme, app=self, page_title="Settings"))
```

The `self.after(500, self._maybe_run_first_startup_train)` scheduling already lives in `V6App.__init__`
behind the `auto_train_on_first_run` flag (Spec 3a). No change needed there.

- [ ] **Step 4:** Run `pytest tests/test_spec3d_settings.py -v` → PASS. Regression sweep → 0 fail.

- [ ] **Step 5:** Commit `feat(spec3d): real SettingsPage (5 working sections) + data-gated auto-train`.

---

## Out of scope (3d)
- Per-metric / per-model sensitivity overrides (single global preset, matching Spec 2). Theme toggle.
  New cleanup categories beyond what V5 has.
</content>
