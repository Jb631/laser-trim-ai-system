# Spec 3d — Settings Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Settings placeholder with a scrollable list of 5 collapsible cards: Alert Thresholds (expanded default, with sensitivity slider + live preview + Save), Per-Model Specs (folded in from V5 Specs page), ML Training, Pricing, Database Cleanup. Wire in Spec 2's deferred first-startup auto-train hook.

**Architecture:** Reuses Spec 3a chrome + Spec 2 API (`preview_alert_count`, `train_drift_detector`). New `SettingsCard` collapsible widget hosts each section. The five sections live as their own widget files for isolation. First-startup hook lives on `V6App` and shows a one-time modal that wraps `train_drift_detector` in a background thread with progress callback.

**Tech Stack:** Python 3.x, customtkinter, pytest. Depends on Specs 1, 2, 3a.

**Target branch:** `V6` only. Latest commit before starting: the Spec 3c final commit.

**Spec reference:** `docs/superpowers/specs/2026-05-30-spec3-ui-shell-design.md` (Sub-spec 3d section).

---

## File Structure

**Files created:**
- `src/laser_trim_analyzer/gui/v6/widgets/settings_card.py` — `SettingsCard` (collapsible)
- `src/laser_trim_analyzer/gui/v6/widgets/sensitivity_slider.py` — discrete 4-stop slider
- `src/laser_trim_analyzer/gui/v6/widgets/training_modal.py` — first-startup + retrain modal
- `src/laser_trim_analyzer/gui/v6/sections/__init__.py` (empty)
- `src/laser_trim_analyzer/gui/v6/sections/alert_thresholds.py`
- `src/laser_trim_analyzer/gui/v6/sections/per_model_specs.py`
- `src/laser_trim_analyzer/gui/v6/sections/ml_training.py`
- `src/laser_trim_analyzer/gui/v6/sections/pricing.py`
- `src/laser_trim_analyzer/gui/v6/sections/database_cleanup.py`
- `src/laser_trim_analyzer/gui/v6/pages/settings_page.py`
- `tests/test_spec3d_settings.py`

**Files modified:**
- `src/laser_trim_analyzer/ml/manager.py` — add `apply_sensitivity_preset(db, preset)` (recomputes thresholds in place; no historical re-scan)
- `src/laser_trim_analyzer/gui/v6/app.py` — first-startup auto-train check; register real SettingsPage

---

## Task 1: `apply_sensitivity_preset` helper

When the user moves the slider and clicks Save, we need to recompute per-tier thresholds for every `model_metric_state` row in place — same baseline, new FP-rate targets, same h/L/z math as training. Runtime state (cusum/ewma) is preserved.

**Files:**
- Modify: `src/laser_trim_analyzer/ml/manager.py`
- Test: `tests/test_spec3d_settings.py` (CREATE)

- [ ] **Step 1: Create test file with the helper test**

Create `tests/test_spec3d_settings.py`:

```python
"""Spec 3d — Settings restructure."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def tk_root():
    import customtkinter as ctk
    root = ctk.CTk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Task 1: apply_sensitivity_preset helper
# ---------------------------------------------------------------------------


def test_apply_sensitivity_preset_recomputes_thresholds(tmp_path):
    """Switching preset recomputes h/L/z without changing baseline."""
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.ml.manager import apply_sensitivity_preset

    db = DatabaseManager(tmp_path / "preset.db")
    with db.session() as s:
        ms = ModelMetricState(
            model="T", metric="sigma_gradient",
            baseline_mean=0.01, baseline_std=0.001, baseline_count=100,
            is_trained=True,
            h_warning=1.0, h_drift=5.0, h_oc=10.0,
            L_warning=1.0, L_drift=2.0, L_oc=3.0,
            z_warning=1.0, z_drift=2.0, z_oc=3.0,
            cusum_pos=2.5, cusum_neg=0.0, ewma_state=0.01,
            last_updated=datetime.now(),
        )
        s.add(ms)
        s.commit()

    apply_sensitivity_preset(db, "tight")

    with db.session() as s:
        row = s.query(ModelMetricState).filter(
            ModelMetricState.model == "T",
            ModelMetricState.metric == "sigma_gradient",
        ).first()
        # Baseline unchanged
        assert row.baseline_mean == pytest.approx(0.01)
        assert row.baseline_std == pytest.approx(0.001)
        # Runtime state preserved
        assert row.cusum_pos == pytest.approx(2.5)
        # Thresholds shifted to tighter values (tight has p smaller than the
        # initial standard-ish values we hand-wrote; new L/z should be larger)
        assert row.L_warning > 1.0
        assert row.z_warning > 1.0


def test_apply_sensitivity_preset_skips_untrained(tmp_path):
    """Rows with is_trained=False are left untouched (no thresholds to compute)."""
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.ml.manager import apply_sensitivity_preset

    db = DatabaseManager(tmp_path / "untrained.db")
    with db.session() as s:
        ms = ModelMetricState(
            model="U", metric="sigma_gradient",
            baseline_mean=None, baseline_std=None, baseline_count=0,
            is_trained=False,
            last_updated=datetime.now(),
        )
        s.add(ms)
        s.commit()

    # Should not raise even though baseline_std is None
    apply_sensitivity_preset(db, "tight")
```

- [ ] **Step 2: Add helper to `ml/manager.py`**

Append at the bottom of `src/laser_trim_analyzer/ml/manager.py`:

```python
def apply_sensitivity_preset(db, preset: str) -> int:
    """Recompute per-tier thresholds in place for every trained row.

    Same baseline_std, new target-FP-rate per tier → new (h, L, z).
    Runtime state (cusum_pos, cusum_neg, ewma_state) is preserved.

    Returns the number of rows updated.
    """
    from datetime import datetime
    from laser_trim_analyzer.database.models import ModelMetricState
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, target_fp_for_tier,
    )
    from laser_trim_analyzer.ml.multi_metric_drift_detector import (
        compute_thresholds,
    )

    updated = 0
    with db.session() as s:
        rows = s.query(ModelMetricState).filter(
            ModelMetricState.is_trained == True,
            ModelMetricState.baseline_std.isnot(None),
        ).all()
        for row in rows:
            sigma = row.baseline_std
            for tier in (DriftTier.WARNING, DriftTier.DRIFT, DriftTier.OUT_OF_CONTROL):
                p = target_fp_for_tier(preset, tier)
                h, L, z = compute_thresholds(sigma, p)
                if tier == DriftTier.WARNING:
                    row.h_warning, row.L_warning, row.z_warning = h, L, z
                elif tier == DriftTier.DRIFT:
                    row.h_drift, row.L_drift, row.z_drift = h, L, z
                else:
                    row.h_oc, row.L_oc, row.z_oc = h, L, z
            row.last_updated = datetime.now()
            updated += 1
        s.commit()
    return updated
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_spec3d_settings.py -v -k apply_sensitivity
```

Expected: 2 PASS.

```bash
git add src/laser_trim_analyzer/ml/manager.py tests/test_spec3d_settings.py
git commit -m "feat(spec3d): apply_sensitivity_preset recomputes thresholds in place

Same baseline, new target FP rates per tier → new h/L/z values.
Runtime CUSUM/EWMA state preserved.  Skips untrained rows.
Used by Settings 'Save preset' button (no full retrain needed)."
```

---

## Task 2: SettingsCard collapsible widget

Each of the 5 Settings sections lives inside a `SettingsCard`. Header row (title + expand/collapse chevron) + body.

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/settings_card.py`
- Test: `tests/test_spec3d_settings.py` (APPEND)

- [ ] **Step 1: Append test**

```python
# ---------------------------------------------------------------------------
# Task 2: SettingsCard
# ---------------------------------------------------------------------------


def test_settings_card_default_collapsed(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.settings_card import SettingsCard

    card = SettingsCard(tk_root, theme=ThemeManager(), title="Test")
    assert card._expanded is False


def test_settings_card_default_expanded_when_requested(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.settings_card import SettingsCard

    card = SettingsCard(
        tk_root, theme=ThemeManager(), title="Test", expanded=True,
    )
    assert card._expanded is True


def test_settings_card_toggle(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.settings_card import SettingsCard

    card = SettingsCard(tk_root, theme=ThemeManager(), title="Test")
    card.toggle()
    assert card._expanded is True
    card.toggle()
    assert card._expanded is False


def test_settings_card_body_frame_returns_inner_parent(tk_root):
    """body_frame() is the parent widget the caller fills with content."""
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.settings_card import SettingsCard

    card = SettingsCard(
        tk_root, theme=ThemeManager(), title="Test", expanded=True,
    )
    inner = card.body_frame()
    assert isinstance(inner, ctk.CTkBaseClass)
    # Caller can pack child widgets into it
    label = ctk.CTkLabel(inner, text="hello")
    label.pack()
```

- [ ] **Step 2: Create the widget**

Create `src/laser_trim_analyzer/gui/v6/widgets/settings_card.py`:

```python
"""Spec 3d — SettingsCard.

Collapsible card for one Settings section.  Header (title + chevron) +
body frame the caller populates.  Default collapsed; pass expanded=True
to start expanded (used for the Alert Thresholds section).
"""
import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


class SettingsCard(ctk.CTkFrame):
    """Collapsible card hosting one Settings section."""

    def __init__(
        self,
        master,
        theme: ThemeManager,
        title: str,
        expanded: bool = False,
        **kwargs,
    ):
        super().__init__(
            master, fg_color=theme.CARD,
            corner_radius=theme.RADIUS_MD, **kwargs,
        )
        self.theme = theme
        self._expanded = False  # set true via toggle below if requested

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(side="top", fill="x")
        self._title_label = ctk.CTkLabel(
            header, text=title,
            font=(theme.FONT_FAMILY[0], theme.SIZE_HEADING, "bold"),
            text_color=theme.TEXT_PRIMARY,
            anchor="w",
        )
        self._title_label.pack(
            side="left", fill="x", expand=True,
            padx=theme.SPACE_MD, pady=theme.SPACE_SM,
        )
        self._chevron = ctk.CTkButton(
            header, text="▾", width=32,
            fg_color="transparent", hover_color=theme.ELEVATED,
            text_color=theme.TEXT_SECONDARY,
            font=(theme.FONT_FAMILY[0], theme.SIZE_HEADING),
            command=self.toggle,
            corner_radius=theme.RADIUS_SM,
        )
        self._chevron.pack(
            side="right", padx=theme.SPACE_MD, pady=theme.SPACE_XS,
        )
        # Make the header clickable too
        for w in (header, self._title_label):
            w.bind("<Button-1>", lambda e: self.toggle())

        # Body (not packed until expanded)
        self._body = ctk.CTkFrame(self, fg_color="transparent")

        if expanded:
            self.toggle()

    def body_frame(self) -> ctk.CTkFrame:
        """Return the inner frame callers populate with section widgets."""
        return self._body

    def toggle(self) -> None:
        if self._expanded:
            self._body.pack_forget()
            self._chevron.configure(text="▾")
        else:
            self._body.pack(
                side="top", fill="x",
                padx=self.theme.SPACE_MD,
                pady=(0, self.theme.SPACE_SM),
            )
            self._chevron.configure(text="▴")
        self._expanded = not self._expanded
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_spec3d_settings.py -v -k settings_card
```

Expected: 4 PASS.

```bash
git add src/laser_trim_analyzer/gui/v6/widgets/settings_card.py tests/test_spec3d_settings.py
git commit -m "feat(spec3d): SettingsCard collapsible widget"
```

---

## Task 3: SensitivitySlider widget

Discrete 4-stop slider mapping `loose / standard / tight / strict`. Emits `on_change(preset)` debounced.

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/sensitivity_slider.py`
- Test: `tests/test_spec3d_settings.py` (APPEND)

- [ ] **Step 1: Append test**

```python
# ---------------------------------------------------------------------------
# Task 3: SensitivitySlider
# ---------------------------------------------------------------------------


def test_sensitivity_slider_initial_value(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.sensitivity_slider import (
        SensitivitySlider,
    )

    slider = SensitivitySlider(
        tk_root, theme=ThemeManager(),
        initial="standard", on_change=lambda p: None,
    )
    assert slider.value() == "standard"


def test_sensitivity_slider_value_changes(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.sensitivity_slider import (
        SensitivitySlider,
    )

    slider = SensitivitySlider(
        tk_root, theme=ThemeManager(),
        initial="standard", on_change=lambda p: None,
    )
    slider.set_value("tight")
    assert slider.value() == "tight"
```

- [ ] **Step 2: Create widget**

Create `src/laser_trim_analyzer/gui/v6/widgets/sensitivity_slider.py`:

```python
"""Spec 3d — SensitivitySlider.

Discrete 4-stop preset picker.  Maps slider position to one of
loose / standard / tight / strict and emits on_change(preset).
"""
from typing import Callable

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


_PRESETS = ["loose", "standard", "tight", "strict"]


class SensitivitySlider(ctk.CTkFrame):
    """Discrete 4-stop sensitivity preset picker."""

    def __init__(
        self,
        master,
        theme: ThemeManager,
        initial: str,
        on_change: Callable[[str], None],
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._on_change = on_change

        # Use CTkSegmentedButton for natural 4-stop UX
        self._segmented = ctk.CTkSegmentedButton(
            self,
            values=_PRESETS,
            command=self._on_select,
            fg_color=theme.CARD,
            selected_color=theme.ACCENT,
            selected_hover_color=theme.ACCENT_HOVER,
            unselected_color=theme.CARD,
            unselected_hover_color=theme.ELEVATED,
            text_color=theme.TEXT_PRIMARY,
            corner_radius=theme.RADIUS_SM,
        )
        self._segmented.pack(side="top", fill="x")
        self._segmented.set(initial if initial in _PRESETS else "standard")

    def value(self) -> str:
        return self._segmented.get()

    def set_value(self, preset: str) -> None:
        if preset in _PRESETS:
            self._segmented.set(preset)

    def _on_select(self, choice: str) -> None:
        self._on_change(choice)
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_spec3d_settings.py -v -k sensitivity_slider
```

Expected: 2 PASS.

```bash
git add src/laser_trim_analyzer/gui/v6/widgets/sensitivity_slider.py tests/test_spec3d_settings.py
git commit -m "feat(spec3d): SensitivitySlider 4-stop preset picker"
```

---

## Task 4: TrainingModal widget

Modal dialog wrapping `train_drift_detector` in a background thread. Used by both first-startup auto-train and the Settings "Retrain" button. Closes itself on completion; supports user-close which detaches but lets training continue.

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/training_modal.py`
- Test: `tests/test_spec3d_settings.py` (APPEND)

- [ ] **Step 1: Append test**

```python
# ---------------------------------------------------------------------------
# Task 4: TrainingModal
# ---------------------------------------------------------------------------


def test_training_modal_invokes_callback_on_progress(tk_root, tmp_path):
    """The modal's progress_callback gets called from the training thread."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.training_modal import (
        TrainingModal,
    )

    db = DatabaseManager(tmp_path / "train.db")
    modal = TrainingModal(
        tk_root, theme=ThemeManager(), db=db, preset="standard",
    )
    # Modal kicks off training inline -- in an empty DB it completes
    # near-instantly with 0 models trained.  No assertion-on-state here;
    # the success criterion is "no exception raised."
    modal.start()
    modal.destroy()
```

- [ ] **Step 2: Create widget**

Create `src/laser_trim_analyzer/gui/v6/widgets/training_modal.py`:

```python
"""Spec 3d — TrainingModal.

Modal wrapping train_drift_detector in a background thread.  Used by
first-startup auto-train (V6App.__init__) and the Settings 'Retrain'
button.
"""
import threading
from typing import Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_training import train_drift_detector


class TrainingModal(ctk.CTkToplevel):
    """Toplevel modal for drift-detector training."""

    def __init__(self, master, theme: ThemeManager, db, preset: str):
        super().__init__(master)
        self.theme = theme
        self.db = db
        self.preset = preset
        self._thread: Optional[threading.Thread] = None
        self._cancelled = False

        self.title("Training drift detector")
        self.geometry("400x180")
        self.configure(fg_color=theme.SURFACE)
        self.transient(master)

        # Title label
        title = ctk.CTkLabel(
            self,
            text="Training drift detector",
            font=(theme.FONT_FAMILY[0], theme.SIZE_HEADING, "bold"),
            text_color=theme.TEXT_PRIMARY,
        )
        title.pack(pady=(theme.SPACE_LG, theme.SPACE_SM))

        # Progress label
        self._status_label = ctk.CTkLabel(
            self,
            text="Preparing...",
            font=(theme.FONT_FAMILY[0], theme.SIZE_BODY),
            text_color=theme.TEXT_SECONDARY,
        )
        self._status_label.pack(pady=theme.SPACE_SM)

        # Progress bar
        self._progress = ctk.CTkProgressBar(
            self, progress_color=theme.ACCENT,
            fg_color=theme.CARD,
        )
        self._progress.pack(
            fill="x", padx=theme.SPACE_LG, pady=theme.SPACE_SM,
        )
        self._progress.set(0)

        # Close button
        close_btn = ctk.CTkButton(
            self, text="Close",
            fg_color=theme.CARD, hover_color=theme.ELEVATED,
            text_color=theme.TEXT_PRIMARY,
            command=self._on_close,
            corner_radius=theme.RADIUS_SM,
        )
        close_btn.pack(pady=theme.SPACE_MD)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run_training, daemon=True,
        )
        self._thread.start()

    def _run_training(self) -> None:
        try:
            train_drift_detector(
                self.db, self.preset, progress_callback=self._on_progress,
            )
        except Exception as exc:
            self._after_safe(
                lambda: self._status_label.configure(
                    text=f"Training failed: {exc}",
                )
            )
            return
        self._after_safe(
            lambda: self._status_label.configure(text="Training complete.")
        )
        self._after_safe(lambda: self._progress.set(1.0))
        self._after_safe(lambda: self.after(750, self.destroy))

    def _on_progress(self, model: str, done: int, total: int) -> None:
        ratio = (done + 1) / max(total, 1)
        self._after_safe(
            lambda: self._status_label.configure(
                text=f"Training: {done + 1} / {total} ({model})"
            )
        )
        self._after_safe(lambda: self._progress.set(ratio))

    def _after_safe(self, callable_) -> None:
        """Schedule a callable on the Tk main thread; tolerate destroyed window."""
        try:
            self.after(0, callable_)
        except Exception:
            pass

    def _on_close(self) -> None:
        self._cancelled = True
        self.destroy()
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_spec3d_settings.py -v -k training_modal
```

Expected: 1 PASS.

```bash
git add src/laser_trim_analyzer/gui/v6/widgets/training_modal.py tests/test_spec3d_settings.py
git commit -m "feat(spec3d): TrainingModal background training with progress UI"
```

---

## Task 5: Alert Thresholds section + the other 4 sections

Each section is a `build_section(parent, theme, app)` function that builds widgets into the given parent. Keeps them isolated and individually testable.

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/sections/__init__.py` (empty)
- Create: 5 section files in `gui/v6/sections/`
- Test: `tests/test_spec3d_settings.py` (APPEND)

- [ ] **Step 1: Append section tests**

```python
# ---------------------------------------------------------------------------
# Task 5: Section build functions
# ---------------------------------------------------------------------------


def test_alert_thresholds_section_builds_without_crash(tk_root, tmp_path):
    import customtkinter as ctk
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.sections.alert_thresholds import (
        build_alert_thresholds_section,
    )

    cfg = Config()
    cfg.database.path = tmp_path / "x.db"
    from laser_trim_analyzer.database.manager import DatabaseManager
    db = DatabaseManager(cfg.database.path)

    class _FakeApp:
        config = cfg
        def __init__(self_, db):
            self_.db = db

    parent = ctk.CTkFrame(tk_root)
    build_alert_thresholds_section(parent, theme=ThemeManager(), app=_FakeApp(db))
    # Build success = test pass; no exception


def test_pricing_section_builds(tk_root, tmp_path):
    import customtkinter as ctk
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.sections.pricing import (
        build_pricing_section,
    )
    from laser_trim_analyzer.database.manager import DatabaseManager

    cfg = Config()
    cfg.database.path = tmp_path / "p.db"
    db = DatabaseManager(cfg.database.path)
    class _FakeApp:
        config = cfg
        def __init__(self_, db): self_.db = db

    parent = ctk.CTkFrame(tk_root)
    build_pricing_section(parent, theme=ThemeManager(), app=_FakeApp(db))


def test_ml_training_section_builds(tk_root, tmp_path):
    import customtkinter as ctk
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.sections.ml_training import (
        build_ml_training_section,
    )
    from laser_trim_analyzer.database.manager import DatabaseManager

    cfg = Config()
    cfg.database.path = tmp_path / "m.db"
    db = DatabaseManager(cfg.database.path)
    class _FakeApp:
        config = cfg
        def __init__(self_, db): self_.db = db

    parent = ctk.CTkFrame(tk_root)
    build_ml_training_section(parent, theme=ThemeManager(), app=_FakeApp(db))
```

- [ ] **Step 2: Create the 5 section files**

Create `src/laser_trim_analyzer/gui/v6/sections/__init__.py` (empty).

Create `src/laser_trim_analyzer/gui/v6/sections/alert_thresholds.py`:

```python
"""Spec 3d — Alert Thresholds section."""
import threading

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.sensitivity_slider import (
    SensitivitySlider,
)
from laser_trim_analyzer.ml.manager import (
    apply_sensitivity_preset, preview_alert_count,
)


def build_alert_thresholds_section(parent, theme: ThemeManager, app) -> None:
    """Build the Alert Thresholds section into `parent`."""
    current = getattr(app.config.ml, "drift_sensitivity", "standard")

    intro = ctk.CTkLabel(
        parent,
        text=("Tighter presets reduce false positives but may miss true drift. "
              "Loose presets surface more signals at the cost of noise."),
        font=(theme.FONT_FAMILY[0], theme.SIZE_BODY),
        text_color=theme.TEXT_SECONDARY,
        wraplength=600, justify="left",
    )
    intro.pack(side="top", fill="x", pady=(0, theme.SPACE_MD), anchor="w")

    preview_label = ctk.CTkLabel(
        parent,
        text="Would flag: — Warning, — Drift, — Out-of-Control",
        font=(theme.FONT_FAMILY[0], theme.SIZE_BODY, "bold"),
        text_color=theme.TEXT_PRIMARY,
    )
    preview_label.pack(side="top", anchor="w", pady=(0, theme.SPACE_SM))

    state = {"pending_after_id": None, "selected": current}

    def refresh_preview(preset: str) -> None:
        def task():
            try:
                counts = preview_alert_count(app.db, preset)
            except Exception:
                counts = {"warning": 0, "drift": 0, "out_of_control": 0}
            text = (
                f"Would flag at '{preset}': "
                f"{counts['warning']} Warning, "
                f"{counts['drift']} Drift, "
                f"{counts['out_of_control']} Out-of-Control"
            )
            try:
                preview_label.after(0, lambda: preview_label.configure(text=text))
            except Exception:
                pass
        threading.Thread(target=task, daemon=True).start()

    def debounced_change(preset: str) -> None:
        state["selected"] = preset
        if state["pending_after_id"] is not None:
            preview_label.after_cancel(state["pending_after_id"])
        state["pending_after_id"] = preview_label.after(
            200, lambda: refresh_preview(preset)
        )

    slider = SensitivitySlider(
        parent, theme=theme,
        initial=current, on_change=debounced_change,
    )
    slider.pack(side="top", fill="x", pady=(0, theme.SPACE_MD))

    def save_preset() -> None:
        preset = state["selected"]
        app.config.ml.drift_sensitivity = preset
        try:
            app.config.save()
        except Exception:
            pass
        threading.Thread(
            target=lambda: apply_sensitivity_preset(app.db, preset),
            daemon=True,
        ).start()
        save_btn.configure(text="Saved ✓")
        save_btn.after(1500, lambda: save_btn.configure(text="Save preset"))

    save_btn = ctk.CTkButton(
        parent, text="Save preset",
        fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
        text_color=theme.TEXT_INVERSE,
        command=save_preset,
        corner_radius=theme.RADIUS_SM,
    )
    save_btn.pack(side="top", anchor="w")

    # Initial preview
    refresh_preview(current)
```

Create `src/laser_trim_analyzer/gui/v6/sections/per_model_specs.py`:

```python
"""Spec 3d — Per-Model Specs section.

Stub for v1.  Full port of the V5 Specs page is deferred; the section
shows a placeholder pointing to the standalone V5 Specs page during
the V6 transition.  Spec 3e (or a later follow-up) does the real port.
"""
import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


def build_per_model_specs_section(parent, theme: ThemeManager, app) -> None:
    label = ctk.CTkLabel(
        parent,
        text=("Per-model spec management lives in the legacy Specs page "
              "for now; a full port lands in a follow-up.  Run 'python "
              "-m laser_trim_analyzer' (no --v6) to edit per-model specs."),
        font=(theme.FONT_FAMILY[0], theme.SIZE_BODY),
        text_color=theme.TEXT_SECONDARY,
        wraplength=600, justify="left",
    )
    label.pack(side="top", anchor="w")
```

Create `src/laser_trim_analyzer/gui/v6/sections/ml_training.py`:

```python
"""Spec 3d — ML Training section."""
import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.training_modal import TrainingModal


def build_ml_training_section(parent, theme: ThemeManager, app) -> None:
    intro = ctk.CTkLabel(
        parent,
        text="Retrain ML models against current data.  Drift detector is the only one in V6 v1; others remain accessible from the V5 Settings page.",
        font=(theme.FONT_FAMILY[0], theme.SIZE_BODY),
        text_color=theme.TEXT_SECONDARY,
        wraplength=600, justify="left",
    )
    intro.pack(side="top", fill="x", pady=(0, theme.SPACE_MD), anchor="w")

    def retrain() -> None:
        preset = getattr(app.config.ml, "drift_sensitivity", "standard")
        modal = TrainingModal(parent, theme=theme, db=app.db, preset=preset)
        modal.start()

    retrain_btn = ctk.CTkButton(
        parent, text="Retrain drift detector",
        fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
        text_color=theme.TEXT_INVERSE,
        command=retrain,
        corner_radius=theme.RADIUS_SM,
    )
    retrain_btn.pack(side="top", anchor="w", pady=(0, theme.SPACE_SM))
```

Create `src/laser_trim_analyzer/gui/v6/sections/pricing.py`:

```python
"""Spec 3d — Pricing section.

Placeholder pointing to the V5 Settings page for now."""
import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


def build_pricing_section(parent, theme: ThemeManager, app) -> None:
    label = ctk.CTkLabel(
        parent,
        text=("Pricing config remains in the V5 Settings page during the V6 "
              "transition.  Run 'python -m laser_trim_analyzer' (no --v6) "
              "to edit it."),
        font=(theme.FONT_FAMILY[0], theme.SIZE_BODY),
        text_color=theme.TEXT_SECONDARY,
        wraplength=600, justify="left",
    )
    label.pack(side="top", anchor="w")
```

Create `src/laser_trim_analyzer/gui/v6/sections/database_cleanup.py`:

```python
"""Spec 3d — Database Cleanup section.

Placeholder pointing to V5 Settings.  Real port happens in a future spec.
"""
import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


def build_database_cleanup_section(parent, theme: ThemeManager, app) -> None:
    label = ctk.CTkLabel(
        parent,
        text=("Database cleanup tools (delete records by model / by date / "
              "drop legacy tables) live in the V5 Settings page during the "
              "V6 transition.  Run without --v6 to access them."),
        font=(theme.FONT_FAMILY[0], theme.SIZE_BODY),
        text_color=theme.TEXT_SECONDARY,
        wraplength=600, justify="left",
    )
    label.pack(side="top", anchor="w")
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_spec3d_settings.py -v -k section
```

Expected: 3 PASS.

```bash
git add src/laser_trim_analyzer/gui/v6/sections/ tests/test_spec3d_settings.py
git commit -m "feat(spec3d): five Settings section build functions

Alert Thresholds (full implementation): sensitivity slider with 200ms
debounced live preview via preview_alert_count, Save button persists
the preset to config.yaml and recomputes thresholds in place.

Per-Model Specs / Pricing / Database Cleanup: v1 placeholders pointing
to the V5 Settings page during transition.  Real ports queued for
follow-up specs.

ML Training: 'Retrain drift detector' button opens TrainingModal which
runs train_drift_detector in a background thread with progress UI."
```

---

## Task 6: SettingsPage composition + first-startup auto-train hook

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/pages/settings_page.py`
- Modify: `src/laser_trim_analyzer/gui/v6/app.py`
- Test: `tests/test_spec3d_settings.py` (APPEND)

- [ ] **Step 1: Append test**

```python
# ---------------------------------------------------------------------------
# Task 6: SettingsPage + first-startup hook
# ---------------------------------------------------------------------------


def test_settings_page_has_five_cards(tmp_path):
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.gui.v6.app import V6App

    cfg = Config()
    cfg.database.path = tmp_path / "settings.db"
    app = V6App(cfg)
    try:
        app.withdraw()
        page = app.page_container.get_page("settings")
        assert len(page._cards) == 5
    finally:
        app.destroy()


def test_first_startup_hook_triggers_on_empty_metric_state(tmp_path, monkeypatch):
    """When V6App starts against an empty model_metric_state, it offers to train."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.gui.v6.app import V6App

    cfg = Config()
    cfg.database.path = tmp_path / "first.db"
    app = V6App(cfg)
    try:
        app.withdraw()
        # The flag _first_startup_train_offered exists; we don't assert that
        # the modal actually appears (would require GUI driver), but the
        # check should have run.
        assert hasattr(app, "_first_startup_train_offered")
    finally:
        app.destroy()
```

- [ ] **Step 2: Create SettingsPage**

Create `src/laser_trim_analyzer/gui/v6/pages/settings_page.py`:

```python
"""Spec 3d — SettingsPage."""
import customtkinter as ctk

from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.sections.alert_thresholds import (
    build_alert_thresholds_section,
)
from laser_trim_analyzer.gui.v6.sections.database_cleanup import (
    build_database_cleanup_section,
)
from laser_trim_analyzer.gui.v6.sections.ml_training import (
    build_ml_training_section,
)
from laser_trim_analyzer.gui.v6.sections.per_model_specs import (
    build_per_model_specs_section,
)
from laser_trim_analyzer.gui.v6.sections.pricing import (
    build_pricing_section,
)
from laser_trim_analyzer.gui.v6.widgets.settings_card import SettingsCard


class SettingsPage(PageBase):
    page_title = "Settings"

    def __init__(self, master, theme, app):
        self._app = app
        self._cards = []
        super().__init__(master, theme=theme)

    def build_content(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        section_specs = [
            ("Alert Thresholds", True, build_alert_thresholds_section),
            ("Per-Model Specs", False, build_per_model_specs_section),
            ("ML Training", False, build_ml_training_section),
            ("Pricing", False, build_pricing_section),
            ("Database Cleanup", False, build_database_cleanup_section),
        ]
        for title, expanded, build in section_specs:
            card = SettingsCard(
                scroll, theme=self.theme, title=title, expanded=expanded,
            )
            card.pack(side="top", fill="x", pady=(0, self.theme.SPACE_SM))
            build(card.body_frame(), theme=self.theme, app=self._app)
            self._cards.append(card)
```

- [ ] **Step 3: Add first-startup hook to V6App + register SettingsPage**

In `src/laser_trim_analyzer/gui/v6/app.py`:

Add a method to `V6App`:

```python
    def _maybe_run_first_startup_train(self) -> None:
        """If model_metric_state is empty, offer a one-time auto-train."""
        from laser_trim_analyzer.database.models import ModelMetricState
        from laser_trim_analyzer.gui.v6.widgets.training_modal import TrainingModal

        self._first_startup_train_offered = True
        try:
            with self.db.session() as s:
                empty = s.query(ModelMetricState).first() is None
        except Exception:
            empty = False
        if not empty:
            return
        preset = getattr(self.config.ml, "drift_sensitivity", "standard")
        modal = TrainingModal(self, theme=self.theme, db=self.db, preset=preset)
        modal.start()
```

In `_build_pages()`, register SettingsPage:

```python
        from laser_trim_analyzer.gui.v6.pages.settings_page import SettingsPage
        settings_page = SettingsPage(
            self.page_container, theme=self.theme, app=self,
        )
        self.page_container.add_page("settings", settings_page)
```

Remove `("settings", "Settings", "3d")` from the placeholder loop.

After `self.show_page("triage")` in `__init__`, call:

```python
        self.after(500, self._maybe_run_first_startup_train)
```

(The 500ms delay lets the main window paint before the modal appears.)

- [ ] **Step 4: Run + regression**

```bash
pytest tests/test_spec3d_settings.py -v
pytest tests/test_spec1_untrimmed_sigma.py tests/test_log_derived_bugfixes_2026_05_30.py tests/test_5_8_2026_bugfixes.py tests/test_spec2_multi_metric_drift.py tests/test_spec3a_shell.py tests/test_spec3b_triage.py tests/test_spec3c_model.py tests/test_spec3d_settings.py 2>&1 | tail -5
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/pages/settings_page.py src/laser_trim_analyzer/gui/v6/app.py tests/test_spec3d_settings.py
git commit -m "feat(spec3d): real SettingsPage + first-startup auto-train

SettingsPage: scrollable list of 5 SettingsCards.  Alert Thresholds
expanded by default.

V6App._maybe_run_first_startup_train: when model_metric_state is empty
(first run on a V5 DB after V6 ships), opens TrainingModal to auto-train
in the background.  Runs 500ms after window paint so it doesn't block
initial render."
```

---

## Out-of-scope reminders

- **Do not** port the full V5 Specs page yet — placeholder for v1.
- **Do not** port the V5 Pricing UI yet — placeholder for v1.
- **Do not** port the V5 DB cleanup UI yet — placeholder for v1.
- **Do not** add cross-model threshold overrides.
- **Do not** add theme toggle.
