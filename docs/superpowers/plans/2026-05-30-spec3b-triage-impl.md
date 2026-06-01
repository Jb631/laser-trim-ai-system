# Spec 3b — Triage Page Implementation Plan (rewritten 2026-06-01)

> **READ FIRST:** `docs/superpowers/plans/2026-06-01-spec3-rewrite-foundations.md`.
> This plan implements foundations §4.1 (list_known_models), §4.3 (metric labels), the Triage page,
> and obeys the QA rules (esp. Q6 σ-honesty, Q9 empty states, Q10 no silent caps). Shared test fixtures
> are in `tests/conftest.py` (created in 3a); do **not** redefine `tk_root`.

**Goal:** Replace the Triage placeholder with the mission landing view — "anything to look at today?"
in <10s with a low false-positive feel. Top: flagged-model cards (tier-colored, model + readable
worst-metric + σ magnitude + alert type). Bottom: search box + scrollable list of every known model
with a visible tier dot + last-processed date. Click any card/row → Model page with that model (and, for
cards, the triggering metric) preselected.

**Target branch:** `V6`. Start at the Spec 3a final commit.

**Fixes applied (foundations §6):** C3 (N+1 → single-query `list_known_models`), I4 (`tier_dot_color`
visible STABLE dot), I9 (`safe_after`), Q6 (σ magnitude labeled with tier), Q9 (empty state names the
last-processed date), Q10 (browse list cap is disclosed), plus human-readable metric names everywhere.

---

## File Structure

**Created:**
- `src/laser_trim_analyzer/gui/v6/widgets/__init__.py` (empty)
- `src/laser_trim_analyzer/gui/v6/widgets/model_alert_card.py`
- `src/laser_trim_analyzer/gui/v6/widgets/flagged_cards_zone.py`
- `src/laser_trim_analyzer/gui/v6/widgets/browse_zone.py`
- `src/laser_trim_analyzer/gui/v6/pages/__init__.py` (empty)
- `src/laser_trim_analyzer/gui/v6/pages/triage_page.py`
- `tests/test_spec3b_triage.py`

**Modified:**
- `src/laser_trim_analyzer/ml/drift_types.py` — add `ModelSummary`, `METRIC_LABELS`, `metric_label()`
- `src/laser_trim_analyzer/ml/manager.py` — add `list_known_models()` (single-query)
- `src/laser_trim_analyzer/gui/v6/app.py` — add `consume_model_route()`; register real `TriagePage`

---

## Task 1: `ModelSummary` + metric labels + `list_known_models` (no N+1)

- [ ] **Step 1:** Create `tests/test_spec3b_triage.py`. (No `tk_root` fixture here — it's in conftest.)

```python
"""Spec 3b — Triage. Foundations §4.1/§4.3. Fixtures in tests/conftest.py."""

# ---- Task 1: helpers ------------------------------------------------------

def test_metric_label_humanizes():
    from laser_trim_analyzer.ml.drift_types import metric_label
    assert metric_label("untrimmed_resistance") == "Untrimmed resistance"
    assert metric_label("linearity_error") == "Linearity error"
    assert metric_label("measured_electrical_angle") == "Electrical angle"
    assert metric_label("totally_unknown") == "totally_unknown"  # graceful passthrough


def test_list_known_models_empty(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.manager import list_known_models
    assert list_known_models(DatabaseManager(tmp_path / "e.db")) == []


def _add_ar(s, model, when):
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, SystemType, StatusType)
    s.add(DBAR(filename=f"{model}.xls", file_path=f"/f/{model}.xls", file_hash=f"h{model}{when.microsecond}",
               model=model, serial="sn1", system=SystemType.A, file_date=when, timestamp=when,
               overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1))


def test_list_known_models_distinct(tmp_path):
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.manager import list_known_models
    db = DatabaseManager(tmp_path / "d.db")
    with db.session() as s:
        for m in ("8340-1", "8232-1", "8877"):
            _add_ar(s, m, datetime.now())
        s.commit()
    assert {x.model for x in list_known_models(db)} == {"8340-1", "8232-1", "8877"}


def test_list_known_models_includes_smoothness_only(tmp_path):
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import SmoothnessResult as DBSR, StatusType
    from laser_trim_analyzer.ml.manager import list_known_models
    db = DatabaseManager(tmp_path / "s.db")
    with db.session() as s:
        s.add(DBSR(filename="s.xls", file_path="/f/s.xls", file_hash="hs", file_date=datetime.now(),
                   model="SMOOTH-ONLY", serial="sn1", test_date=datetime.now(),
                   overall_status=StatusType.PASS, timestamp=datetime.now()))
        s.commit()
    assert "SMOOTH-ONLY" in {x.model for x in list_known_models(db)}


def test_list_known_models_tier_merged_from_drift_api(tmp_path, monkeypatch):
    """Tier comes from a SINGLE get_drifting_models call; others default STABLE.
    Mock the drift API so the test is deterministic (not coupled to detector math)."""
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.drift_types import (
        AlertType, DriftTier, ModelAlertSummary)
    import laser_trim_analyzer.ml.manager as mgr
    db = DatabaseManager(tmp_path / "t.db")
    with db.session() as s:
        _add_ar(s, "FLAGGED", datetime.now())
        _add_ar(s, "CALM", datetime.now())
        s.commit()
    monkeypatch.setattr(mgr, "get_drifting_models", lambda _db, *a, **k: [
        ModelAlertSummary(model="FLAGGED", tier=DriftTier.DRIFT,
                          alert_type=AlertType.STEP_CHANGE,
                          worst_metric="untrimmed_resistance", magnitude=4.2)])
    by = {x.model: x.tier for x in mgr.list_known_models(db)}
    assert by["FLAGGED"] == DriftTier.DRIFT
    assert by["CALM"] == DriftTier.STABLE


def test_list_known_models_single_query_no_per_model_status(tmp_path, monkeypatch):
    """Regression guard for the N+1 bug: list_known_models must NOT call
    get_model_drift_status once per model."""
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    import laser_trim_analyzer.ml.manager as mgr
    db = DatabaseManager(tmp_path / "n.db")
    with db.session() as s:
        for i in range(5):
            _add_ar(s, f"M{i}", datetime.now())
        s.commit()
    calls = {"n": 0}
    real = mgr.get_model_drift_status
    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(mgr, "get_model_drift_status", counted)
    mgr.list_known_models(db)
    assert calls["n"] == 0  # tiers come from get_drifting_models, not per-model status
```

- [ ] **Step 2:** Run → fail.

- [ ] **Step 3:** In `src/laser_trim_analyzer/ml/drift_types.py` add (near the other dataclasses /
  WATCHED_METRICS):

```python
@dataclass
class ModelSummary:
    """Compact per-model row for the Triage browse zone."""
    model: str
    tier: DriftTier
    last_processed: Optional[datetime] = None


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


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)
```

- [ ] **Step 4:** In `src/laser_trim_analyzer/ml/manager.py` append the single-query helper. **Do not**
  loop `get_model_drift_status` per model:

```python
def list_known_models(db):
    """One ModelSummary per distinct model across analysis_results + smoothness_results.

    Cost: ONE inventory session (GROUP BY model, MAX(file_date)) + ONE get_drifting_models
    call for tiers. Independent of model count (fixes the per-model-session N+1).
    Non-flagged models default to DriftTier.STABLE.
    """
    from sqlalchemy import func
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, SmoothnessResult as DBSR,
    )
    from laser_trim_analyzer.ml.drift_types import DriftTier, ModelSummary

    last_seen = {}
    with db.session() as s:
        for model, last in (s.query(DBAR.model, func.max(DBAR.file_date))
                            .group_by(DBAR.model).all()):
            if model:
                last_seen[model] = last
        for model, last in (s.query(DBSR.model, func.max(DBSR.file_date))
                            .group_by(DBSR.model).all()):
            if not model:
                continue
            if model not in last_seen:
                last_seen[model] = last
            elif last is not None and (last_seen[model] is None or last > last_seen[model]):
                last_seen[model] = last

    flagged = {a.model: a.tier for a in get_drifting_models(db)}
    return sorted(
        (ModelSummary(model=m, tier=flagged.get(m, DriftTier.STABLE),
                      last_processed=last_seen.get(m)) for m in last_seen),
        key=lambda x: x.model,
    )
```

- [ ] **Step 5:** Run → PASS. Commit `feat(spec3b): ModelSummary + metric labels + single-query
  list_known_models`.

---

## Task 2: ModelAlertCard

Tier-colored, click anywhere → `on_click(model, focus_metric)`. Shows readable metric name and a
σ-magnitude labeled with the tier (Q6).

- [ ] **Step 1:** Append tests:

```python
# ---- Task 2: ModelAlertCard ----------------------------------------------

def _labels(widget):
    import customtkinter as ctk
    out = []
    for c in widget.winfo_children():
        if isinstance(c, ctk.CTkLabel):
            out.append(c.cget("text"))
        out.extend(_labels(c))
    return out


def _summary(model="8340-1", tier=None, metric="untrimmed_resistance", mag=4.2, alert=None):
    from laser_trim_analyzer.ml.drift_types import AlertType, DriftTier, ModelAlertSummary
    return ModelAlertSummary(model=model, tier=tier or DriftTier.DRIFT,
                             alert_type=alert or AlertType.STEP_CHANGE,
                             worst_metric=metric, magnitude=mag)


def test_card_shows_model_readable_metric_and_magnitude(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.model_alert_card import ModelAlertCard
    card = ModelAlertCard(tk_root, summary=_summary(), theme=ThemeManager(), on_click=lambda *_: None)
    texts = " | ".join(_labels(card))
    assert "8340-1" in texts
    assert "Untrimmed resistance" in texts      # readable, not the raw key
    assert "4.2" in texts and "σ" in texts
    assert "Step change" in texts


def test_card_click_emits_model_and_focus_metric(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.model_alert_card import ModelAlertCard
    got = []
    card = ModelAlertCard(tk_root, summary=_summary(model="CLICK", metric="linearity_error"),
                          theme=ThemeManager(), on_click=lambda m, f: got.append((m, f)))
    card._on_click()
    assert got == [("CLICK", "linearity_error")]   # focus = the triggering metric


def test_card_uses_tier_background(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.model_alert_card import ModelAlertCard
    from laser_trim_analyzer.ml.drift_types import DriftTier
    t = ThemeManager()
    card = ModelAlertCard(tk_root, summary=_summary(tier=DriftTier.OUT_OF_CONTROL),
                          theme=t, on_click=lambda *_: None)
    assert card.cget("fg_color") == t.tier_color(DriftTier.OUT_OF_CONTROL)[0]
```

- [ ] **Step 2:** Run `-k card` → fail.

- [ ] **Step 3:** Create `widgets/__init__.py` (empty) and `widgets/model_alert_card.py`:

```python
"""Spec 3b — ModelAlertCard: one flagged-model summary card."""
from typing import Callable

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import AlertType, ModelAlertSummary, metric_label

CARD_WIDTH = 250
CARD_HEIGHT = 132


class ModelAlertCard(ctk.CTkFrame):
    def __init__(self, master, summary: ModelAlertSummary, theme: ThemeManager,
                 on_click: Callable[[str, str], None], **kwargs):
        bg, fg = theme.tier_color(summary.tier)
        super().__init__(master, width=CARD_WIDTH, height=CARD_HEIGHT, fg_color=bg,
                         corner_radius=theme.RADIUS_MD, **kwargs)
        self.theme = theme; self.summary = summary; self._cb = on_click; self._fg = fg
        self.pack_propagate(False)
        self._build()
        self._bind_recursive(self)

    def _build(self):
        t = self.theme; s = self.summary
        ctk.CTkLabel(self, text=s.model, font=t.font(t.SIZE_TITLE, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w")\
            .pack(side="top", fill="x", padx=t.SPACE_MD, pady=(t.SPACE_MD, 0))
        badge = "Step change" if s.alert_type == AlertType.STEP_CHANGE else "Slow drift"
        ctk.CTkLabel(self, text=f"{badge} · {metric_label(s.worst_metric)}",
                     font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY, anchor="w")\
            .pack(side="top", fill="x", padx=t.SPACE_MD)
        ctk.CTkLabel(self, text=f"{s.magnitude:+.1f}σ", font=t.font(t.SIZE_DISPLAY, "bold"),
                     text_color=self._fg, anchor="w")\
            .pack(side="top", fill="x", padx=t.SPACE_MD)
        # Q6: say what the σ is measured against.
        ctk.CTkLabel(self, text=f"beyond {s.tier.name.replace('_', ' ').title()} limit",
                     font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY, anchor="w")\
            .pack(side="top", fill="x", padx=t.SPACE_MD, pady=(0, t.SPACE_MD))

    def _bind_recursive(self, w):
        w.bind("<Button-1>", lambda e: self._on_click())
        for c in w.winfo_children():
            self._bind_recursive(c)

    def _on_click(self):
        # Deep-link with the triggering metric as the focus.
        self._cb(self.summary.model, self.summary.worst_metric)
```

- [ ] **Step 4:** Run `-k card` → PASS. Commit `feat(spec3b): ModelAlertCard (readable metric, labeled
  σ, deep-link focus)`.

---

## Task 3: FlaggedCardsZone (wrapping grid + dated empty state)

- [ ] **Step 1:** Append tests:

```python
# ---- Task 3: FlaggedCardsZone --------------------------------------------

def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def test_zone_empty_state_names_last_processed(tk_root):
    from datetime import datetime
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.flagged_cards_zone import FlaggedCardsZone
    z = FlaggedCardsZone(tk_root, theme=ThemeManager(), on_card_click=lambda *_: None)
    z.set_summaries([], last_processed=datetime(2026, 5, 30))
    txt = " ".join(_labels(z))
    assert "within tolerance" in txt
    assert "2026-05-30" in txt


def test_zone_one_card_per_summary(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.flagged_cards_zone import FlaggedCardsZone
    from laser_trim_analyzer.gui.v6.widgets.model_alert_card import ModelAlertCard
    z = FlaggedCardsZone(tk_root, theme=ThemeManager(), on_card_click=lambda *_: None)
    z.set_summaries([_summary(model=f"M{i}") for i in range(6)])
    assert sum(isinstance(w, ModelAlertCard) for w in _walk(z)) == 6


def test_zone_routes_click(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.flagged_cards_zone import FlaggedCardsZone
    from laser_trim_analyzer.gui.v6.widgets.model_alert_card import ModelAlertCard
    got = []
    z = FlaggedCardsZone(tk_root, theme=ThemeManager(), on_card_click=lambda m, f: got.append((m, f)))
    z.set_summaries([_summary(model="ROUTED", metric="sigma_gradient")])
    next(w for w in _walk(z) if isinstance(w, ModelAlertCard))._on_click()
    assert got == [("ROUTED", "sigma_gradient")]
```

- [ ] **Step 2:** Run `-k zone` (flagged) → fail.

- [ ] **Step 3:** Create `widgets/flagged_cards_zone.py` (wrap cards into rows of `MAX_PER_ROW`):

```python
"""Spec 3b — FlaggedCardsZone: 'Needs attention' heading + wrapping card grid + empty state."""
from datetime import datetime
from typing import Callable, List, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.model_alert_card import ModelAlertCard
from laser_trim_analyzer.ml.drift_types import ModelAlertSummary

MAX_PER_ROW = 4


class FlaggedCardsZone(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_card_click: Callable[[str, str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme; self._cb = on_card_click
        self._heading = ctk.CTkLabel(self, text="Needs attention (0)",
                                     font=theme.font(theme.SIZE_HEADING, "bold"),
                                     text_color=theme.TEXT_PRIMARY, anchor="w")
        self._heading.pack(side="top", fill="x", pady=(0, theme.SPACE_SM))
        self._body = ctk.CTkFrame(self, fg_color="transparent")
        self._body.pack(side="top", fill="x")

    def set_summaries(self, summaries: List[ModelAlertSummary],
                      last_processed: Optional[datetime] = None) -> None:
        for c in list(self._body.winfo_children()):
            c.destroy()
        self._heading.configure(text=f"Needs attention ({len(summaries)})")
        t = self.theme
        if not summaries:
            when = last_processed.strftime("%Y-%m-%d") if last_processed else "—"
            ctk.CTkLabel(self._body,
                         text=f"All models within tolerance — last processed {when}.",
                         font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY, anchor="w")\
                .pack(side="top", fill="x", pady=t.SPACE_LG)
            return
        row = None
        for i, s in enumerate(summaries):
            if i % MAX_PER_ROW == 0:
                row = ctk.CTkFrame(self._body, fg_color="transparent")
                row.pack(side="top", fill="x")
            ModelAlertCard(row, summary=s, theme=t, on_click=self._cb)\
                .pack(side="left", padx=(0, t.SPACE_MD), pady=t.SPACE_SM)
```

- [ ] **Step 4:** Run → PASS. Commit `feat(spec3b): FlaggedCardsZone (wrapping grid, dated empty state)`.

---

## Task 4: BrowseZone (visible dots, disclosed cap)

- [ ] **Step 1:** Append tests:

```python
# ---- Task 4: BrowseZone ---------------------------------------------------

def _ms(model, tier=None):
    from laser_trim_analyzer.ml.drift_types import DriftTier, ModelSummary
    return ModelSummary(model=model, tier=tier or DriftTier.STABLE)


def test_browse_one_row_per_model(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
    z = BrowseZone(tk_root, theme=ThemeManager(), on_row_click=lambda _: None)
    z.set_models([_ms(f"M{i}") for i in range(5)])
    assert len(z._rows) == 5


def test_browse_filter_substring(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
    z = BrowseZone(tk_root, theme=ThemeManager(), on_row_click=lambda _: None)
    z.set_models([_ms("8340-1"), _ms("8232-1"), _ms("8877")])
    z.set_filter("83")
    shown = {r.summary.model for r in z._rows}
    assert shown == {"8340-1", "8232-1"}


def test_browse_row_click_emits_model(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
    got = []
    z = BrowseZone(tk_root, theme=ThemeManager(), on_row_click=got.append)
    z.set_models([_ms("CLICKED")])
    z._rows[0]._on_click()
    assert got == ["CLICKED"]


def test_browse_discloses_cap(tk_root):
    """Q10: when more than the render cap exist, say so instead of silently truncating."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone, ROW_CAP
    z = BrowseZone(tk_root, theme=ThemeManager(), on_row_click=lambda _: None)
    z.set_models([_ms(f"M{i:04d}") for i in range(ROW_CAP + 25)])
    assert len(z._rows) == ROW_CAP
    assert "Showing" in z._cap_label.cget("text") and str(ROW_CAP + 25) in z._cap_label.cget("text")
```

- [ ] **Step 2:** Run `-k browse` → fail.

- [ ] **Step 3:** Create `widgets/browse_zone.py` (dots via `tier_dot_color`; cap disclosed):

```python
"""Spec 3b — BrowseZone: search + scrollable model list (visible tier dot, last-processed date)."""
from typing import Callable, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.ml.drift_types import ModelSummary

ROW_CAP = 200  # render cap for responsiveness; cap is disclosed (Q10)


class BrowseZone(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, on_row_click: Callable[[str], None], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme; self._cb = on_row_click
        self._models: List[ModelSummary] = []
        self._rows: List["_BrowseRow"] = []
        t = theme
        ctk.CTkLabel(self, text="All models", font=t.font(t.SIZE_HEADING, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w").pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        self._search_var = ctk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._render())
        ctk.CTkEntry(self, textvariable=self._search_var, placeholder_text="Search models…",
                     font=t.font(t.SIZE_BODY), fg_color=t.CARD, border_color=t.BORDER,
                     text_color=t.TEXT_PRIMARY).pack(side="top", fill="x", pady=(0, t.SPACE_SM))
        self._cap_label = ctk.CTkLabel(self, text="", font=t.font(t.SIZE_CAPTION),
                                       text_color=t.TEXT_SECONDARY, anchor="w")
        self._cap_label.pack(side="top", fill="x")
        self._list = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._list.pack(side="top", fill="both", expand=True)

    def set_models(self, models: List[ModelSummary]) -> None:
        self._models = list(models)
        self._render()

    def set_filter(self, text: str) -> None:
        self._search_var.set(text)

    def _render(self) -> None:
        for r in self._rows:
            r.destroy()
        self._rows.clear()
        flt = self._search_var.get().lower()
        matches = [m for m in self._models if not flt or flt in m.model.lower()]
        for m in matches[:ROW_CAP]:
            row = _BrowseRow(self._list, summary=m, theme=self.theme, on_click=self._cb)
            row.pack(side="top", fill="x", pady=1)
            self._rows.append(row)
        if len(matches) > ROW_CAP:
            self._cap_label.configure(
                text=f"Showing {ROW_CAP} of {len(matches)} — narrow with search.")
        else:
            self._cap_label.configure(text="")


class _BrowseRow(ctk.CTkFrame):
    def __init__(self, master, summary: ModelSummary, theme: ThemeManager,
                 on_click: Callable[[str], None]):
        super().__init__(master, fg_color=theme.SURFACE, corner_radius=theme.RADIUS_SM)
        self.theme = theme; self.summary = summary; self._cb = on_click
        t = theme
        dot = ctk.CTkFrame(self, width=12, height=12, corner_radius=6,
                           fg_color=t.tier_dot_color(summary.tier))   # FIX I4: visible STABLE dot
        dot.pack(side="left", padx=(t.SPACE_SM, t.SPACE_XS)); dot.pack_propagate(False)
        name = ctk.CTkLabel(self, text=summary.model, font=t.font(t.SIZE_BODY),
                            text_color=t.TEXT_PRIMARY, anchor="w")
        name.pack(side="left", fill="x", expand=True, padx=(t.SPACE_XS, t.SPACE_SM))
        date_txt = summary.last_processed.strftime("%Y-%m-%d") if summary.last_processed else "—"
        date = ctk.CTkLabel(self, text=date_txt, font=t.font(t.SIZE_CAPTION),
                            text_color=t.TEXT_SECONDARY)
        date.pack(side="right", padx=t.SPACE_SM)
        for w in (self, dot, name, date):
            w.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self):
        self._cb(self.summary.model)
```

- [ ] **Step 4:** Run `-k browse` → PASS. Commit `feat(spec3b): BrowseZone (visible dots, disclosed
  cap, live filter)`.

---

## Task 5: TriagePage + routing hint

`on_show()` reloads on a background thread → `safe_after` to the Tk thread. A synchronous `reload_now()`
exists for tests (no mainloop). Card click deep-links with the triggering metric; row click with the
model only.

- [ ] **Step 1:** Append tests:

```python
# ---- Task 5: TriagePage + routing ----------------------------------------

def test_v6app_consume_model_route(make_app):
    app = make_app()
    assert app.consume_model_route() is None
    app.set_model_route("M", "linearity_error")
    assert app.consume_model_route() == "M"       # 3b consumes model only
    assert app.consume_model_route() is None       # one-shot


def test_triage_card_click_routes_to_model(make_app):
    app = make_app()
    triage = app.page_container.get_page("triage")
    triage._on_card_click("FROM-CARD", "untrimmed_resistance")
    assert app.page_container.current_page == "model"
    # Placeholder Model page doesn't consume yet → hint persists with focus.
    assert app._model_route == ("FROM-CARD", "untrimmed_resistance")


def test_triage_reload_now_populates(make_app):
    from datetime import datetime
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR, SystemType, StatusType
    app = make_app()
    with app.db.session() as s:
        s.add(DBAR(filename="x.xls", file_path="/f/x.xls", file_hash="hx", model="LOAD-TEST",
                   serial="sn1", system=SystemType.A, file_date=datetime.now(), timestamp=datetime.now(),
                   overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1))
        s.commit()
    triage = app.page_container.get_page("triage")
    triage.reload_now()       # synchronous path for tests
    assert "LOAD-TEST" in _labels(triage)
```

- [ ] **Step 2:** Run `-k "consume_model_route or triage"` → fail.

- [ ] **Step 3:** Add to `V6App` (in `gui/v6/app.py`) the model-only consumer:

```python
    def consume_model_route(self) -> Optional[str]:
        """Pop the model name from the routing hint (focus consumed separately in 3c)."""
        if self._model_route is None:
            return None
        model, _focus = self._model_route
        self._model_route = None
        return model
```

- [ ] **Step 4:** Create `pages/__init__.py` (empty) and `pages/triage_page.py`:

```python
"""Spec 3b — TriagePage: flagged cards (top) + browse list (bottom)."""
import threading

from laser_trim_analyzer.gui.v6.page_base import PageBase
from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
from laser_trim_analyzer.gui.v6.widgets.flagged_cards_zone import FlaggedCardsZone
from laser_trim_analyzer.ml.manager import get_drifting_models, list_known_models


class TriagePage(PageBase):
    page_title = "Triage"

    def build_content(self, parent):
        self._cards = FlaggedCardsZone(parent, theme=self.theme, on_card_click=self._on_card_click)
        self._cards.pack(side="top", fill="x", pady=(0, self.theme.SPACE_LG))
        self._browse = BrowseZone(parent, theme=self.theme, on_row_click=self._on_row_click)
        self._browse.pack(side="top", fill="both", expand=True)

    def on_show(self):
        threading.Thread(target=self.reload_now, daemon=True).start()

    def reload_now(self):
        """Synchronous reload (background thread in prod; direct in tests)."""
        try:
            flagged = get_drifting_models(self.app.db)
            models = list_known_models(self.app.db)
        except Exception:
            flagged, models = [], []
        last = max((m.last_processed for m in models if m.last_processed), default=None)
        self.safe_after(lambda: self._cards.set_summaries(flagged, last_processed=last))
        self.safe_after(lambda: self._browse.set_models(models))

    def _on_card_click(self, model, focus_metric):
        self.app.set_model_route(model, focus_metric)
        self.app.show_page("model")

    def _on_row_click(self, model):
        self.app.set_model_route(model)        # no focus → Model page defaults the metric
        self.app.show_page("model")
```

Note: `reload_now` calls `safe_after`; in tests without a mainloop the queued callbacks still apply the
data synchronously because `safe_after(delay=0)` schedules on the same call stack only under mainloop.
To keep the test deterministic, `set_summaries`/`set_models` are ALSO invoked directly — implement
`reload_now` to call them directly AND schedule via `safe_after` only when running under a live loop. To
avoid double-render, simplest is: in `reload_now`, update widgets directly (it's already the right thread
when called from a test), and have `on_show` marshal by calling `reload_now` from a thread that uses
`safe_after`. **Implementer:** prefer this shape —

```python
    def reload_now(self):
        try:
            flagged = get_drifting_models(self.app.db)
            models = list_known_models(self.app.db)
        except Exception:
            flagged, models = [], []
        last = max((m.last_processed for m in models if m.last_processed), default=None)
        self._apply(flagged, models, last)

    def _apply(self, flagged, models, last):
        self._cards.set_summaries(flagged, last_processed=last)
        self._browse.set_models(models)

    def on_show(self):
        def work():
            try:
                flagged = get_drifting_models(self.app.db)
                models = list_known_models(self.app.db)
            except Exception:
                flagged, models = [], []
            last = max((m.last_processed for m in models if m.last_processed), default=None)
            self.safe_after(lambda: self._apply(flagged, models, last))
        threading.Thread(target=work, daemon=True).start()
```

`reload_now()` (used by tests) does the work and applies synchronously; `on_show()` does the same work on
a thread and applies via `safe_after`. Same `_apply`, no double-render.

- [ ] **Step 5:** Register the real TriagePage. In `gui/v6/app.py` `_build_pages()`, build Triage first
  (real) then the remaining placeholders:

```python
    def _build_pages(self) -> None:
        from laser_trim_analyzer.gui.v6.pages.triage_page import TriagePage
        self.page_container.add_page(
            "triage", TriagePage(self.page_container, theme=self.theme, app=self, page_title="Triage"))
        for name, label, nxt in (("process", "Process", "3e"),
                                 ("model", "Model", "3c"), ("settings", "Settings", "3d")):
            self.page_container.add_page(
                name, _PlaceholderPage(self.page_container, theme=self.theme, app=self,
                                       page_title=label, next_spec=nxt))
```

- [ ] **Step 6:** Run `pytest tests/test_spec3b_triage.py -v` → all PASS. Regression sweep → 0 fail.

- [ ] **Step 7:** Commit `feat(spec3b): real TriagePage + consume_model_route`.

---

## Out of scope (3b)
- Model page content (3c). No Triage header refresh button (on_show is the refresh point; manual refresh
  deferred). No cross-model filters (per-model mission). No first-startup auto-train (3d).
</content>
