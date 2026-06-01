# Spec 3b — Triage Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Triage placeholder page with the real landing view: flagged-cards zone on top (3-5 ModelAlertCards showing model + worst metric + magnitude + alert type, tier-colored), browse zone on bottom (search box + scrollable model list). Click any card or row → Model page with that model preselected.

**Architecture:** New widgets under `gui/v6/widgets/`. The Triage page composes `FlaggedCardsZone` + `BrowseZone`. Data flows via `ml.manager.get_drifting_models()` (Spec 2 public API) and a new `list_known_models()` helper. Refresh happens in `on_show()` via background thread; UI updates on main thread.

**Tech Stack:** Python 3.x, customtkinter, pytest. Depends on Spec 2's drift detector + public API and Spec 3a's shell.

**Target branch:** `V6` only. Latest commit before starting: the Spec 3a final commit.

**Spec reference:** `docs/superpowers/specs/2026-05-30-spec3-ui-shell-design.md` (Sub-spec 3b section).

---

## File Structure

**Files created:**
- `src/laser_trim_analyzer/gui/v6/widgets/__init__.py` (empty package marker)
- `src/laser_trim_analyzer/gui/v6/widgets/model_alert_card.py` — `ModelAlertCard`
- `src/laser_trim_analyzer/gui/v6/widgets/flagged_cards_zone.py` — `FlaggedCardsZone`
- `src/laser_trim_analyzer/gui/v6/widgets/browse_zone.py` — `BrowseZone`
- `src/laser_trim_analyzer/gui/v6/pages/__init__.py` (empty package marker)
- `src/laser_trim_analyzer/gui/v6/pages/triage_page.py` — `TriagePage`
- `tests/test_spec3b_triage.py`

**Files modified:**
- `src/laser_trim_analyzer/ml/manager.py` — add `list_known_models()` helper
- `src/laser_trim_analyzer/gui/v6/app.py` — replace TriagePlaceholder with real `TriagePage`

---

## Task 1: `list_known_models` helper in ml/manager.py

Adds a single function that returns all models with at least one record in either `analysis_results` or `smoothness_results` plus a tier for each. Sorted by model name. Used by the BrowseZone.

**Files:**
- Modify: `src/laser_trim_analyzer/ml/manager.py`
- Test: `tests/test_spec3b_triage.py` (CREATE)

- [ ] **Step 1: Create the test file with the helper test**

Create `tests/test_spec3b_triage.py`:

```python
"""Spec 3b — Triage page (landing view).

Each test maps to one element of the spec at
docs/superpowers/specs/2026-05-30-spec3-ui-shell-design.md (Sub-spec 3b).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# Task 1: list_known_models helper
# ---------------------------------------------------------------------------


def test_list_known_models_returns_empty_for_empty_db(tmp_path):
    """Fresh DB → empty list."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.manager import list_known_models

    db = DatabaseManager(tmp_path / "empty.db")
    assert list_known_models(db) == []


def test_list_known_models_returns_distinct_models(tmp_path):
    """Returns one summary per distinct model seen in analysis_results."""
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, SystemType as DBSystemType,
        StatusType as DBStatusType,
    )
    from laser_trim_analyzer.ml.manager import list_known_models

    db = DatabaseManager(tmp_path / "models.db")
    with db.session() as s:
        for model_name in ("8340-1", "8232-1", "8877"):
            ar = DBAR(
                filename=f"{model_name}-test.xls",
                file_path=f"/fake/{model_name}.xls",
                file_hash=f"h-{model_name}",
                model=model_name,
                serial="sn1",
                system=DBSystemType.A,
                file_date=datetime.now(),
                timestamp=datetime.now(),
                overall_status=DBStatusType.PASS,
                has_multi_tracks=False,
                processing_time=0.1,
            )
            s.add(ar)
        s.commit()

    summaries = list_known_models(db)
    model_names = {s.model for s in summaries}
    assert model_names == {"8340-1", "8232-1", "8877"}


def test_list_known_models_includes_smoothness_only_models(tmp_path):
    """Models that have only smoothness records also appear."""
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        SmoothnessResult as DBSR, StatusType as DBStatusType,
    )
    from laser_trim_analyzer.ml.manager import list_known_models

    db = DatabaseManager(tmp_path / "smooth.db")
    with db.session() as s:
        sr = DBSR(
            filename="smooth-only.xls",
            file_path="/fake/smooth-only.xls",
            file_hash="hsm1",
            file_date=datetime.now(),
            model="SMOOTH-ONLY",
            serial="sn1",
            test_date=datetime.now(),
            overall_status=DBStatusType.PASS,
            timestamp=datetime.now(),
        )
        s.add(sr)
        s.commit()

    summaries = list_known_models(db)
    assert "SMOOTH-ONLY" in {x.model for x in summaries}


def test_list_known_models_carries_tier_from_metric_state(tmp_path):
    """Each summary's tier reflects the model's current state per
    get_model_drift_status (worst-of across metrics).
    """
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, ModelMetricState,
        SystemType as DBSystemType, StatusType as DBStatusType,
    )
    from laser_trim_analyzer.ml.drift_types import DriftTier
    from laser_trim_analyzer.ml.manager import list_known_models

    db = DatabaseManager(tmp_path / "tiered.db")
    today = datetime.now()
    with db.session() as s:
        ar = DBAR(
            filename="t.xls", file_path="/fake/t.xls",
            file_hash="ht", model="TIERED", serial="sn1",
            system=DBSystemType.A, file_date=today, timestamp=today,
            overall_status=DBStatusType.PASS, has_multi_tracks=False,
            processing_time=0.1,
        )
        s.add(ar)
        # Hand-write a metric state row that puts TIERED into Warning
        ms = ModelMetricState(
            model="TIERED", metric="sigma_gradient",
            baseline_mean=0.01, baseline_std=0.001, baseline_count=100,
            is_trained=True,
            h_warning=1.0, h_drift=5.0, h_oc=10.0,
            L_warning=2.0, L_drift=3.0, L_oc=4.0,
            z_warning=1.6, z_drift=2.3, z_oc=3.0,
            cusum_pos=2.0, cusum_neg=0.0, ewma_state=0.01,
            last_updated=today,
        )
        s.add(ms)
        s.commit()

    summaries = list_known_models(db)
    tiered = next(x for x in summaries if x.model == "TIERED")
    assert tiered.tier >= DriftTier.WARNING
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_spec3b_triage.py -v`

Expected: 4 FAILs — `list_known_models` not found, and `ModelSummary` dataclass not defined.

- [ ] **Step 3: Add the helper + `ModelSummary` dataclass**

In `src/laser_trim_analyzer/ml/drift_types.py`, add a new dataclass at the bottom:

```python
@dataclass
class ModelSummary:
    """Compact form for Spec 3 Triage browse zone."""
    model: str
    tier: DriftTier
    last_processed: Optional[datetime] = None
```

In `src/laser_trim_analyzer/ml/manager.py`, append at the bottom:

```python
def list_known_models(db):
    """Return one ModelSummary per distinct model in the DB.

    Combines models from analysis_results and smoothness_results.
    Each summary's tier reflects the model's current drift state
    (worst-of) per get_model_drift_status.  Sorted by model name.

    For Spec 3 Triage browse zone.
    """
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR,
        SmoothnessResult as DBSR,
    )
    from laser_trim_analyzer.ml.drift_types import ModelSummary

    with db.session() as s:
        trim_models = [r[0] for r in s.query(DBAR.model).distinct().all()]
        smooth_models = [r[0] for r in s.query(DBSR.model).distinct().all()]

    all_models = sorted(set(trim_models) | set(smooth_models))
    summaries = []
    for model in all_models:
        status = get_model_drift_status(db, model)
        summaries.append(ModelSummary(
            model=model,
            tier=status.overall_tier,
            last_processed=status.last_processed,
        ))
    return summaries
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_spec3b_triage.py -v`

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/ml/drift_types.py src/laser_trim_analyzer/ml/manager.py tests/test_spec3b_triage.py
git commit -m "feat(spec3b): list_known_models + ModelSummary

Helper that returns one summary per distinct model in the DB
(trim or smoothness), each with its current drift tier from
get_model_drift_status.  Used by Triage browse zone.

Adds the ModelSummary dataclass to drift_types alongside the
existing summaries."
```

---

## Task 2: ModelAlertCard widget

A tier-colored card showing model + worst metric + magnitude + alert type. Click → emits `on_click(model)`.

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/__init__.py` (empty)
- Create: `src/laser_trim_analyzer/gui/v6/widgets/model_alert_card.py`
- Test: `tests/test_spec3b_triage.py` (APPEND)

- [ ] **Step 1: Append the ModelAlertCard tests**

```python
# ---------------------------------------------------------------------------
# Task 2: ModelAlertCard widget
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tk_root():
    """Module-scoped headless CTk root."""
    import customtkinter as ctk
    root = ctk.CTk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


def test_model_alert_card_renders_summary_fields(tk_root):
    """Card displays model, worst_metric, magnitude, alert_type text."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.model_alert_card import (
        ModelAlertCard,
    )
    from laser_trim_analyzer.ml.drift_types import (
        AlertType, DriftTier, ModelAlertSummary,
    )

    summary = ModelAlertSummary(
        model="8340-1",
        tier=DriftTier.DRIFT,
        alert_type=AlertType.STEP_CHANGE,
        worst_metric="untrimmed_resistance",
        magnitude=4.2,
    )
    card = ModelAlertCard(
        tk_root, summary=summary, theme=ThemeManager(),
        on_click=lambda model: None,
    )
    # Inspect the labels the card built
    texts = _collect_label_texts(card)
    assert "8340-1" in texts
    assert "untrimmed_resistance" in texts
    assert any("4.2" in t for t in texts)
    assert any("Step" in t for t in texts)


def test_model_alert_card_click_emits_model_name(tk_root):
    """Clicking the card fires on_click(model)."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.model_alert_card import (
        ModelAlertCard,
    )
    from laser_trim_analyzer.ml.drift_types import (
        AlertType, DriftTier, ModelAlertSummary,
    )

    received: list[str] = []
    summary = ModelAlertSummary(
        model="TEST-CLICK", tier=DriftTier.WARNING,
        alert_type=AlertType.SLOW_DRIFT,
        worst_metric="sigma_gradient", magnitude=1.5,
    )
    card = ModelAlertCard(
        tk_root, summary=summary, theme=ThemeManager(),
        on_click=lambda model: received.append(model),
    )
    card._on_click()  # simulate
    assert received == ["TEST-CLICK"]


def test_model_alert_card_uses_tier_background(tk_root):
    """Card's fg_color matches the spec's tier_color background."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.model_alert_card import (
        ModelAlertCard,
    )
    from laser_trim_analyzer.ml.drift_types import (
        AlertType, DriftTier, ModelAlertSummary,
    )

    theme = ThemeManager()
    summary = ModelAlertSummary(
        model="X", tier=DriftTier.OUT_OF_CONTROL,
        alert_type=AlertType.STEP_CHANGE,
        worst_metric="sigma_gradient", magnitude=5.0,
    )
    card = ModelAlertCard(
        tk_root, summary=summary, theme=theme,
        on_click=lambda model: None,
    )
    bg, _ = theme.tier_color(DriftTier.OUT_OF_CONTROL)
    assert card.cget("fg_color") == bg


def _collect_label_texts(widget):
    """Walk a widget tree and return all CTkLabel texts as a list."""
    import customtkinter as ctk

    out = []
    for child in widget.winfo_children():
        if isinstance(child, ctk.CTkLabel):
            out.append(child.cget("text"))
        out.extend(_collect_label_texts(child))
    return out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_spec3b_triage.py -v -k model_alert_card`

Expected: 3 FAILs.

- [ ] **Step 3: Create the widget**

Create `src/laser_trim_analyzer/gui/v6/widgets/__init__.py` (empty).

Create `src/laser_trim_analyzer/gui/v6/widgets/model_alert_card.py`:

```python
"""Spec 3b — ModelAlertCard widget.

A clickable tier-colored card summarizing one flagged model.  Used by
FlaggedCardsZone on the Triage page.
"""
from typing import Callable

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import (
    AlertType, ModelAlertSummary,
)


CARD_WIDTH: int = 240
CARD_HEIGHT: int = 120


class ModelAlertCard(ctk.CTkFrame):
    """One flagged-model summary card."""

    def __init__(
        self,
        master,
        summary: ModelAlertSummary,
        theme: ThemeManager,
        on_click: Callable[[str], None],
        **kwargs,
    ):
        bg, fg = theme.tier_color(summary.tier)
        super().__init__(
            master,
            width=CARD_WIDTH,
            height=CARD_HEIGHT,
            fg_color=bg,
            corner_radius=theme.RADIUS_MD,
            **kwargs,
        )
        self.theme = theme
        self.summary = summary
        self._on_click_external = on_click
        self._fg = fg

        self.pack_propagate(False)
        self._build()
        self._bind_click_recursive(self)

    # ---- Construction ----------------------------------------------------

    def _build(self) -> None:
        # Model name (large)
        model_label = ctk.CTkLabel(
            self,
            text=self.summary.model,
            font=(self.theme.FONT_FAMILY[0], self.theme.SIZE_TITLE, "bold"),
            text_color=self.theme.TEXT_PRIMARY,
            anchor="w",
        )
        model_label.pack(
            side="top", fill="x",
            padx=self.theme.SPACE_MD, pady=(self.theme.SPACE_MD, 0),
        )

        # Worst metric subtitle
        metric_label = ctk.CTkLabel(
            self,
            text=self.summary.worst_metric,
            font=(self.theme.FONT_FAMILY[0], self.theme.SIZE_CAPTION),
            text_color=self.theme.TEXT_SECONDARY,
            anchor="w",
        )
        metric_label.pack(
            side="top", fill="x", padx=self.theme.SPACE_MD,
        )

        # Magnitude (large, tier accent color)
        magnitude_text = f"{self.summary.magnitude:+.1f}σ"
        magnitude_label = ctk.CTkLabel(
            self,
            text=magnitude_text,
            font=(self.theme.FONT_FAMILY[0], self.theme.SIZE_DISPLAY, "bold"),
            text_color=self._fg,
            anchor="w",
        )
        magnitude_label.pack(
            side="top", fill="x", padx=self.theme.SPACE_MD,
        )

        # Alert type badge
        badge_text = (
            "Step change"
            if self.summary.alert_type == AlertType.STEP_CHANGE
            else "Slow drift"
        )
        badge_label = ctk.CTkLabel(
            self,
            text=badge_text,
            font=(self.theme.FONT_FAMILY[0], self.theme.SIZE_CAPTION, "bold"),
            text_color=self.theme.TEXT_PRIMARY,
            anchor="w",
        )
        badge_label.pack(
            side="top", fill="x",
            padx=self.theme.SPACE_MD, pady=(0, self.theme.SPACE_MD),
        )

    # ---- Click handling --------------------------------------------------

    def _bind_click_recursive(self, widget) -> None:
        """Make the entire card clickable, not just the outer frame."""
        widget.bind("<Button-1>", lambda e: self._on_click())
        for child in widget.winfo_children():
            self._bind_click_recursive(child)

    def _on_click(self) -> None:
        self._on_click_external(self.summary.model)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_spec3b_triage.py -v -k model_alert_card`

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/widgets/__init__.py src/laser_trim_analyzer/gui/v6/widgets/model_alert_card.py tests/test_spec3b_triage.py
git commit -m "feat(spec3b): ModelAlertCard widget

Tier-colored card showing model name, worst metric, magnitude (large
accent-colored), and alert-type badge.  Entire card is clickable (binds
recursively) and emits on_click(model)."
```

---

## Task 3: FlaggedCardsZone widget

Holds the "Needs attention" section label + a grid of ModelAlertCards. Shows an empty-state message when nothing is flagged.

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/flagged_cards_zone.py`
- Test: `tests/test_spec3b_triage.py` (APPEND)

- [ ] **Step 1: Append FlaggedCardsZone tests**

```python
# ---------------------------------------------------------------------------
# Task 3: FlaggedCardsZone widget
# ---------------------------------------------------------------------------


def test_flagged_cards_zone_empty_state(tk_root):
    """With no summaries, zone shows the 'all within tolerance' message."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.flagged_cards_zone import (
        FlaggedCardsZone,
    )

    zone = FlaggedCardsZone(
        tk_root, theme=ThemeManager(), on_card_click=lambda m: None,
    )
    zone.set_summaries([])
    texts = _collect_label_texts(zone)
    assert any("All models within tolerance" in t for t in texts)


def test_flagged_cards_zone_renders_card_per_summary(tk_root):
    """Each summary → one ModelAlertCard child."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.flagged_cards_zone import (
        FlaggedCardsZone,
    )
    from laser_trim_analyzer.gui.v6.widgets.model_alert_card import (
        ModelAlertCard,
    )
    from laser_trim_analyzer.ml.drift_types import (
        AlertType, DriftTier, ModelAlertSummary,
    )

    summaries = [
        ModelAlertSummary(
            model=f"M{i}", tier=DriftTier.WARNING,
            alert_type=AlertType.STEP_CHANGE,
            worst_metric="sigma_gradient", magnitude=2.0 + i,
        )
        for i in range(3)
    ]
    zone = FlaggedCardsZone(
        tk_root, theme=ThemeManager(), on_card_click=lambda m: None,
    )
    zone.set_summaries(summaries)
    cards = [w for w in _walk(zone) if isinstance(w, ModelAlertCard)]
    assert len(cards) == 3


def test_flagged_cards_zone_routes_card_click_to_callback(tk_root):
    """Clicking a card surfaces the model name to the zone's on_card_click."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.flagged_cards_zone import (
        FlaggedCardsZone,
    )
    from laser_trim_analyzer.gui.v6.widgets.model_alert_card import (
        ModelAlertCard,
    )
    from laser_trim_analyzer.ml.drift_types import (
        AlertType, DriftTier, ModelAlertSummary,
    )

    received: list[str] = []
    zone = FlaggedCardsZone(
        tk_root, theme=ThemeManager(),
        on_card_click=lambda m: received.append(m),
    )
    zone.set_summaries([
        ModelAlertSummary(
            model="ROUTED", tier=DriftTier.DRIFT,
            alert_type=AlertType.SLOW_DRIFT,
            worst_metric="linearity_error", magnitude=3.0,
        ),
    ])
    cards = [w for w in _walk(zone) if isinstance(w, ModelAlertCard)]
    cards[0]._on_click()
    assert received == ["ROUTED"]


def _walk(widget):
    """Yield all descendants of a widget (including the widget itself)."""
    yield widget
    for child in widget.winfo_children():
        yield from _walk(child)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_spec3b_triage.py -v -k flagged_cards`

Expected: 3 FAILs.

- [ ] **Step 3: Create the FlaggedCardsZone**

Create `src/laser_trim_analyzer/gui/v6/widgets/flagged_cards_zone.py`:

```python
"""Spec 3b — FlaggedCardsZone.

Top section of the Triage page.  Shows a 'Needs attention (N)' label
and a horizontal grid of ModelAlertCards.  Empty state when nothing
is flagged.
"""
from typing import Callable, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.model_alert_card import (
    ModelAlertCard,
)
from laser_trim_analyzer.ml.drift_types import ModelAlertSummary


class FlaggedCardsZone(ctk.CTkFrame):
    """Top zone of Triage page: flagged-model alert cards."""

    def __init__(
        self,
        master,
        theme: ThemeManager,
        on_card_click: Callable[[str], None],
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._on_card_click = on_card_click

        # Section heading
        self._heading = ctk.CTkLabel(
            self,
            text="Needs attention (0)",
            font=(theme.FONT_FAMILY[0], theme.SIZE_HEADING, "bold"),
            text_color=theme.TEXT_PRIMARY,
            anchor="w",
        )
        self._heading.pack(
            side="top", fill="x", pady=(0, theme.SPACE_SM),
        )

        # Container for cards (or empty-state label)
        self._cards_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._cards_frame.pack(side="top", fill="x")

    def set_summaries(self, summaries: List[ModelAlertSummary]) -> None:
        """Replace current cards with new ones derived from summaries."""
        # Clear existing children
        for child in self._cards_frame.winfo_children():
            child.destroy()

        # Update heading
        self._heading.configure(text=f"Needs attention ({len(summaries)})")

        if not summaries:
            empty = ctk.CTkLabel(
                self._cards_frame,
                text="All models within tolerance — no drift detected.",
                font=(self.theme.FONT_FAMILY[0], self.theme.SIZE_BODY),
                text_color=self.theme.TEXT_SECONDARY,
                anchor="w",
            )
            empty.pack(side="top", fill="x", pady=self.theme.SPACE_LG)
            return

        # Lay out cards left-to-right with wrap (manual since CTk doesn't
        # have a built-in flow layout).  For v1: simple horizontal row.
        row = ctk.CTkFrame(self._cards_frame, fg_color="transparent")
        row.pack(side="top", fill="x")
        for summary in summaries:
            card = ModelAlertCard(
                row, summary=summary, theme=self.theme,
                on_click=self._on_card_click,
            )
            card.pack(
                side="left", padx=(0, self.theme.SPACE_MD),
                pady=self.theme.SPACE_SM,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_spec3b_triage.py -v -k flagged_cards`

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/widgets/flagged_cards_zone.py tests/test_spec3b_triage.py
git commit -m "feat(spec3b): FlaggedCardsZone

Holds the 'Needs attention (N)' heading + horizontal row of
ModelAlertCards.  Empty-state message when nothing is flagged.
set_summaries(list) replaces the current cards."
```

---

## Task 4: BrowseZone widget

Search box + scrollable list of `ModelSummary` rows. Each row shows a tier-color dot + model name. Click → emits `on_row_click(model)`. Search filter is case-insensitive substring.

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/browse_zone.py`
- Test: `tests/test_spec3b_triage.py` (APPEND)

- [ ] **Step 1: Append BrowseZone tests**

```python
# ---------------------------------------------------------------------------
# Task 4: BrowseZone widget
# ---------------------------------------------------------------------------


def test_browse_zone_renders_one_row_per_model(tk_root):
    """Each model in set_models becomes a row."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, ModelSummary,
    )

    models = [
        ModelSummary(model=f"M{i}", tier=DriftTier.STABLE)
        for i in range(5)
    ]
    zone = BrowseZone(
        tk_root, theme=ThemeManager(), on_row_click=lambda m: None,
    )
    zone.set_models(models)
    texts = _collect_label_texts(zone)
    for m in models:
        assert m.model in texts


def test_browse_zone_filter_substring(tk_root):
    """Typing in the search filters the displayed rows."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, ModelSummary,
    )

    models = [
        ModelSummary(model="8340-1", tier=DriftTier.STABLE),
        ModelSummary(model="8232-1", tier=DriftTier.STABLE),
        ModelSummary(model="8877", tier=DriftTier.STABLE),
    ]
    zone = BrowseZone(
        tk_root, theme=ThemeManager(), on_row_click=lambda m: None,
    )
    zone.set_models(models)
    zone.set_filter("83")
    texts = _collect_label_texts(zone)
    assert "8340-1" in texts
    assert "8232-1" in texts
    assert "8877" not in texts


def test_browse_zone_row_click_emits_model(tk_root):
    """Clicking a row fires on_row_click(model_name)."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
    from laser_trim_analyzer.ml.drift_types import (
        DriftTier, ModelSummary,
    )

    received: list[str] = []
    zone = BrowseZone(
        tk_root, theme=ThemeManager(),
        on_row_click=lambda m: received.append(m),
    )
    zone.set_models([
        ModelSummary(model="CLICKED", tier=DriftTier.STABLE),
    ])
    # Find the row and click
    zone._rows[0]._on_click()
    assert received == ["CLICKED"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_spec3b_triage.py -v -k browse_zone`

Expected: 3 FAILs.

- [ ] **Step 3: Create the BrowseZone**

Create `src/laser_trim_analyzer/gui/v6/widgets/browse_zone.py`:

```python
"""Spec 3b — BrowseZone.

Bottom section of the Triage page.  Search box + scrollable list of
all models.  Each row: tier color dot + model name + last-processed
date.  Click → on_row_click(model).
"""
from typing import Callable, List, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import (
    DriftTier, ModelSummary,
)


class BrowseZone(ctk.CTkFrame):
    """Bottom zone of Triage: search + scrollable model list."""

    def __init__(
        self,
        master,
        theme: ThemeManager,
        on_row_click: Callable[[str], None],
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._on_row_click = on_row_click
        self._models: List[ModelSummary] = []
        self._filter: str = ""
        self._rows: List["_BrowseRow"] = []

        self._build()

    # ---- Construction ----------------------------------------------------

    def _build(self) -> None:
        # Heading
        heading = ctk.CTkLabel(
            self,
            text="All models",
            font=(self.theme.FONT_FAMILY[0], self.theme.SIZE_HEADING, "bold"),
            text_color=self.theme.TEXT_PRIMARY,
            anchor="w",
        )
        heading.pack(side="top", fill="x", pady=(0, self.theme.SPACE_SM))

        # Search input
        self._search_var = ctk.StringVar()
        self._search_var.trace_add("write", self._on_search_change)
        search_entry = ctk.CTkEntry(
            self,
            textvariable=self._search_var,
            placeholder_text="Search models...",
            font=(self.theme.FONT_FAMILY[0], self.theme.SIZE_BODY),
            fg_color=self.theme.CARD,
            border_color=self.theme.BORDER,
            text_color=self.theme.TEXT_PRIMARY,
        )
        search_entry.pack(
            side="top", fill="x", pady=(0, self.theme.SPACE_SM),
        )

        # Scrollable list
        self._list_frame = ctk.CTkScrollableFrame(
            self, fg_color="transparent",
        )
        self._list_frame.pack(side="top", fill="both", expand=True)

    # ---- Public API ------------------------------------------------------

    def set_models(self, models: List[ModelSummary]) -> None:
        """Replace the current model list."""
        self._models = list(models)
        self._render()

    def set_filter(self, text: str) -> None:
        """Programmatic filter (mainly for tests)."""
        self._search_var.set(text)

    # ---- Internal --------------------------------------------------------

    def _on_search_change(self, *args) -> None:
        self._filter = self._search_var.get().lower()
        self._render()

    def _render(self) -> None:
        # Clear existing rows
        for row in self._rows:
            row.destroy()
        self._rows.clear()
        # Re-render filtered
        for m in self._models:
            if self._filter and self._filter not in m.model.lower():
                continue
            row = _BrowseRow(
                self._list_frame, summary=m, theme=self.theme,
                on_click=self._on_row_click,
            )
            row.pack(side="top", fill="x", pady=1)
            self._rows.append(row)


class _BrowseRow(ctk.CTkFrame):
    """One model row inside the browse list."""

    def __init__(
        self,
        master,
        summary: ModelSummary,
        theme: ThemeManager,
        on_click: Callable[[str], None],
    ):
        super().__init__(
            master, fg_color=theme.SURFACE, corner_radius=theme.RADIUS_SM,
        )
        self.theme = theme
        self.summary = summary
        self._on_click_external = on_click

        # Tier color dot (12x12 frame)
        dot_bg, _ = theme.tier_color(summary.tier)
        dot = ctk.CTkFrame(
            self, width=12, height=12, fg_color=dot_bg,
            corner_radius=6,
        )
        dot.pack(side="left", padx=(theme.SPACE_SM, theme.SPACE_XS))
        dot.pack_propagate(False)

        # Model name
        name_label = ctk.CTkLabel(
            self, text=summary.model,
            font=(theme.FONT_FAMILY[0], theme.SIZE_BODY),
            text_color=theme.TEXT_PRIMARY,
            anchor="w",
        )
        name_label.pack(side="left", fill="x", expand=True,
                        padx=(theme.SPACE_XS, theme.SPACE_SM))

        # Last-processed date (right-anchored)
        date_text = (
            summary.last_processed.strftime("%Y-%m-%d")
            if summary.last_processed else "—"
        )
        date_label = ctk.CTkLabel(
            self, text=date_text,
            font=(theme.FONT_FAMILY[0], theme.SIZE_CAPTION),
            text_color=theme.TEXT_SECONDARY,
        )
        date_label.pack(side="right", padx=theme.SPACE_SM)

        # Make the whole row clickable
        for w in (self, name_label, dot, date_label):
            w.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self) -> None:
        self._on_click_external(self.summary.model)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_spec3b_triage.py -v -k browse_zone`

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/widgets/browse_zone.py tests/test_spec3b_triage.py
git commit -m "feat(spec3b): BrowseZone

Search + scrollable list of all models.  Each row: tier color dot,
model name, last-processed date.  Case-insensitive substring filter
updates live.  Clicking a row emits on_row_click(model)."
```

---

## Task 5: TriagePage + routing hint

`TriagePage` composes `FlaggedCardsZone` + `BrowseZone`. `on_show()` reloads from the DB (background thread). Card and row clicks call `V6App.set_model_route(model)` + `V6App.show_page("model")`. Spec 3a's `V6App` gets two new methods.

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/pages/__init__.py` (empty)
- Create: `src/laser_trim_analyzer/gui/v6/pages/triage_page.py`
- Modify: `src/laser_trim_analyzer/gui/v6/app.py` — add routing-hint methods + register real TriagePage
- Test: `tests/test_spec3b_triage.py` (APPEND)

- [ ] **Step 1: Append integration tests**

```python
# ---------------------------------------------------------------------------
# Task 5: TriagePage + V6App routing hint
# ---------------------------------------------------------------------------


def test_v6app_exposes_routing_hint(tmp_path):
    """V6App has set_model_route / consume_model_route methods."""
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.gui.v6.app import V6App

    cfg = Config()
    cfg.database.path = tmp_path / "route.db"
    app = V6App(cfg)
    try:
        app.withdraw()
        # No hint set
        assert app.consume_model_route() is None
        # Set + consume
        app.set_model_route("MY-MODEL")
        assert app.consume_model_route() == "MY-MODEL"
        # Already consumed
        assert app.consume_model_route() is None
    finally:
        app.destroy()


def test_triage_page_card_click_routes_to_model_page(tmp_path, monkeypatch):
    """Clicking a flagged card stashes the model on the app and navigates
    to the Model page.
    """
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.gui.v6.app import V6App

    cfg = Config()
    cfg.database.path = tmp_path / "route2.db"
    app = V6App(cfg)
    try:
        app.withdraw()
        triage = app.page_container.get_page("triage")
        # Synthetic click handler
        triage._on_card_click("FROM-CARD")
        assert app.page_container.current_page == "model"
        # Consumer hasn't run yet (placeholder page) so the hint persists
        assert app.consume_model_route() == "FROM-CARD"
    finally:
        app.destroy()


def test_triage_page_on_show_loads_data(tmp_path):
    """on_show() populates the flagged cards + browse zone from the DB.

    Synchronous load for this test (background-thread path covered by
    integration tests in a future spec; this verifies the data path).
    """
    from datetime import datetime
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR,
        SystemType as DBSystemType, StatusType as DBStatusType,
    )
    from laser_trim_analyzer.gui.v6.app import V6App

    cfg = Config()
    cfg.database.path = tmp_path / "loaded.db"
    app = V6App(cfg)
    try:
        app.withdraw()
        # Add a model
        with app.db.session() as s:
            ar = DBAR(
                filename="x.xls", file_path="/fake/x.xls", file_hash="hx",
                model="LOAD-TEST", serial="sn1",
                system=DBSystemType.A, file_date=datetime.now(),
                timestamp=datetime.now(),
                overall_status=DBStatusType.PASS,
                has_multi_tracks=False, processing_time=0.1,
            )
            s.add(ar)
            s.commit()

        triage = app.page_container.get_page("triage")
        # Force synchronous reload for the test
        triage._reload_sync()
        texts = _collect_label_texts(triage)
        assert "LOAD-TEST" in texts
    finally:
        app.destroy()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_spec3b_triage.py -v -k "routing_hint or triage_page"`

Expected: 3 FAILs.

- [ ] **Step 3: Add routing-hint methods to V6App**

In `src/laser_trim_analyzer/gui/v6/app.py`, add inside `V6App` before `_build_pages`:

```python
    # ---- Routing-hint API (consumed by Model page on show) ---------------

    def set_model_route(self, model: str, focus_metric: Optional[str] = None) -> None:
        """Stash a routing hint that the Model page reads on next on_show.

        Used by Triage and search-bar navigation.
        """
        self._model_route = (model, focus_metric)

    def consume_model_route(self) -> Optional[str]:
        """Pop the stashed model name.  Returns None if no hint set.

        Note: the focus_metric (Spec 3c) is consumed by a separate method.
        For Spec 3b only the model name matters.
        """
        if self._model_route is None:
            return None
        model, _focus = self._model_route
        self._model_route = None
        return model
```

Initialize `self._model_route = None` at the top of `V6App.__init__` (just after `self.theme = ThemeManager()`).

- [ ] **Step 4: Create the TriagePage**

Create `src/laser_trim_analyzer/gui/v6/pages/__init__.py` (empty).

Create `src/laser_trim_analyzer/gui/v6/pages/triage_page.py`:

```python
"""Spec 3b — TriagePage.

Landing view.  Flagged-cards zone on top + browse zone on bottom.
on_show() reloads data from the DB.  Card or row click → navigate
to the Model page with the model preselected.
"""
import threading
from typing import Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
from laser_trim_analyzer.gui.v6.widgets.flagged_cards_zone import (
    FlaggedCardsZone,
)
from laser_trim_analyzer.ml.manager import (
    get_drifting_models, list_known_models,
)


class TriagePage(PageBase):
    """Triage landing page.

    Refresh model: on_show() runs the DB queries in a background thread
    and updates the UI on the main thread via self.after(0, ...).
    """
    page_title = "Triage"

    def __init__(self, master, theme, app):
        # `app` is the V6App reference, needed for routing-hint + show_page
        self._app = app
        super().__init__(master, theme=theme)

    def build_content(self, parent):
        self._cards_zone = FlaggedCardsZone(
            parent, theme=self.theme, on_card_click=self._on_card_click,
        )
        self._cards_zone.pack(
            side="top", fill="x", pady=(0, self.theme.SPACE_LG),
        )
        self._browse_zone = BrowseZone(
            parent, theme=self.theme, on_row_click=self._on_card_click,
        )
        self._browse_zone.pack(side="top", fill="both", expand=True)

    def on_show(self):
        """Reload data asynchronously so a slow DB doesn't block the UI."""
        thread = threading.Thread(target=self._reload_sync, daemon=True)
        thread.start()

    def _reload_sync(self) -> None:
        """Synchronous reload.  Used by background thread AND by tests."""
        try:
            sensitivity = getattr(
                self._app.config.ml, "drift_sensitivity", "standard"
            )
            flagged = get_drifting_models(self._app.db, sensitivity)
            all_models = list_known_models(self._app.db)
        except Exception:
            # Defensive: if DB queries fail we don't crash the UI
            flagged = []
            all_models = []
        # UI updates must happen on the main thread
        self.after(0, lambda: self._cards_zone.set_summaries(flagged))
        self.after(0, lambda: self._browse_zone.set_models(all_models))

    def _on_card_click(self, model: str) -> None:
        self._app.set_model_route(model)
        self._app.show_page("model")
```

- [ ] **Step 5: Replace TriagePlaceholder in V6App**

In `src/laser_trim_analyzer/gui/v6/app.py`, find `_build_pages()`. Replace the loop with:

```python
    def _build_pages(self) -> None:
        """Construct all 4 pages once.  Triage is real (Spec 3b);
        others are placeholders until 3c/3d/3e land."""
        from laser_trim_analyzer.gui.v6.pages.triage_page import TriagePage

        triage = TriagePage(self.page_container, theme=self.theme, app=self)
        self.page_container.add_page("triage", triage)

        for name, label, next_spec in (
            ("process", "Process", "3e"),
            ("model", "Model", "3c"),
            ("settings", "Settings", "3d"),
        ):
            page = _PlaceholderPage(
                self.page_container, theme=self.theme,
                page_title=label, next_spec=next_spec,
            )
            self.page_container.add_page(name, page)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_spec3b_triage.py -v`

Expected: all PASS (~16 tests total: 4 + 3 + 3 + 3 + 3).

- [ ] **Step 7: Regression sweep**

```
pytest tests/test_spec1_untrimmed_sigma.py tests/test_log_derived_bugfixes_2026_05_30.py tests/test_5_8_2026_bugfixes.py tests/test_spec2_multi_metric_drift.py tests/test_spec3a_shell.py tests/test_spec3b_triage.py -v 2>&1 | tail -5
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/app.py src/laser_trim_analyzer/gui/v6/pages/__init__.py src/laser_trim_analyzer/gui/v6/pages/triage_page.py tests/test_spec3b_triage.py
git commit -m "feat(spec3b): real TriagePage replaces placeholder

Composes FlaggedCardsZone (top) + BrowseZone (bottom).  on_show()
reloads data in a background thread; UI updates on main thread.
Card + row clicks stash the model name via V6App.set_model_route
and navigate to the Model page (which will read the hint when 3c
ships).

Adds set_model_route / consume_model_route API to V6App."
```

---

## Out-of-scope reminders

- **Do not** implement the Model page real content (3c).
- **Do not** add a refresh button to the Triage header (deferred — `on_show` is the refresh point).
- **Do not** add cross-model filters in BrowseZone (mission is per-model).
- **Do not** wire the first-startup auto-train hook (3d).
