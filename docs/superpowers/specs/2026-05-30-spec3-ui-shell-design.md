# Spec 3 — V6 UI Shell Design (Umbrella)

**Date:** 2026-05-30
**Status:** Approved, ready for implementation plans
**Parent specs:**
- `docs/superpowers/specs/2026-05-14-app-mission-realignment-design.md` (anchor)
- `docs/superpowers/specs/2026-05-30-app-redesign-design.md` (V6 umbrella)
**Depends on:** Spec 1 (`untrimmed_sigma_gradient`), Spec 2 (multi-metric drift detector + public API)
**Target branch:** `V6` only.
**Source:** Brainstorming session with James 2026-05-30 (after Spec 2 implementation landed).

Spec 3 builds the **new V6 UI shell + 4 pages** that consume the data and APIs from Specs 1 and 2. The V6 design doc described 4 top-level pages (Triage / Process / Model / Settings) with old pages (Dashboard, Analyze, Compare, Trends, Quality Health, Scorecard, Specs, Smoothness) absorbed or retired.

The work decomposes into **5 sub-specs**. Each gets its own implementation plan and execution cycle:

| Sub-spec | Scope | Estimated effort |
|---|---|---|
| **3a** | App shell + sidebar nav + theme system + PageBase + 4 placeholder pages | Small (~8 TDD tasks) |
| **3b** | Triage page — flagged cards + browse zone | Medium (~10 tasks) |
| **3c** | Model page — focus chart + 8-pill glance + tabs (Drift / Smoothness / Units) | Large (~15 tasks) |
| **3d** | Settings restructure — sensitivity slider + 5 collapsible sections + first-startup auto-train | Medium (~12 tasks) |
| **3e** | Process page rewrite + retire old V5 pages + promote `gui/v6/` to `gui/` | Medium (~10 tasks) |

---

## Cross-cutting decisions (apply to all 5 sub-specs)

### Tech stack

- **Framework:** customtkinter, unchanged from V5. Existing chart widget (`gui/widgets/chart.py`) and form patterns transfer directly.
- **No web frontend.** Mission is per-station desktop app, deployed as a Windows .exe.
- **No PyQt migration.** Out of scope.

### Theme — industrial dark

Single dark palette. All colors, fonts, spacing, and corner radii defined as constants in `gui/v6/theme.py`. No magic numbers in page or widget code.

**Color tokens:**

```python
# Surfaces
BG          = "#1a1f2e"   # outermost background
SURFACE     = "#1e2435"   # page content surface
CARD        = "#263244"   # cards/panels on top of SURFACE
ELEVATED    = "#2f3b50"   # hover / focused card

# Sidebar
SIDEBAR_BG     = "#1a1f2e"
SIDEBAR_ACTIVE = "#263244"
SIDEBAR_STRIPE = "#3b82f6"

# Accent
ACCENT          = "#3b82f6"
ACCENT_HOVER    = "#60a5fa"
ACCENT_PRESSED  = "#2563eb"

# Text
TEXT_PRIMARY    = "#e8eef5"
TEXT_SECONDARY  = "#9ca8bd"
TEXT_DISABLED   = "#5a6478"
TEXT_INVERSE    = "#1a1f2e"

# Borders / dividers
DIVIDER         = "#2a3142"
BORDER          = "#3a4456"
```

**Tier colors (preserved semantic from V5):**

```python
TIER_STABLE     = "#1e2435"   # blends with SURFACE
TIER_WARNING_BG = "#3d2f1a"
TIER_WARNING    = "#f59e0b"
TIER_DRIFT_BG   = "#3d2418"
TIER_DRIFT      = "#f97316"
TIER_OOC_BG     = "#3d1818"
TIER_OOC        = "#ef4444"
```

**Typography:**

```python
FONT_FAMILY = ("Inter", "Segoe UI", "system-ui")

SIZE_CAPTION    = 11
SIZE_BODY       = 13
SIZE_HEADING    = 16
SIZE_TITLE      = 20
SIZE_DISPLAY    = 28
```

**Spacing scale (multiples of 4):** `SPACE_XS=4`, `SPACE_SM=8`, `SPACE_MD=12`, `SPACE_LG=16`, `SPACE_XL=24`, `SPACE_2XL=32`.

**Corner radii:** `RADIUS_SM=4`, `RADIUS_MD=6`, `RADIUS_LG=8`.

### Coexistence model during development

- Spec 3a creates everything new under `src/laser_trim_analyzer/gui/v6/`.
- Old V5 UI (`gui/app.py`, `gui/pages/*`) stays intact and runnable until Spec 3e retires it in one cleanup commit.
- `python -m laser_trim_analyzer.app` runs the V5 shell. `python -m laser_trim_analyzer.app --v6` runs the V6 shell. Same DB.
- After Spec 3e:
  - Delete `gui/app.py`, all `gui/pages/*` files, all 8 retired pages
  - Promote `gui/v6/*` up to `gui/` (i.e., `gui/v6/app.py` → `gui/app.py`)
  - Delete the `--v6` CLI flag
  - V6 is the only UI

### Old V5 pages — fate per page

| V5 page | Fate |
|---|---|
| Dashboard | Deleted; replaced by Triage |
| Process | Rewritten in 3e using V6 patterns; deleted from `gui/pages/` after promotion |
| Analyze | Deleted; per-unit drill-down absorbed into Model page's Units tab |
| Compare | Deleted; baseline-vs-recent absorbed into Model page focus chart |
| Trends | Deleted; per-model historical lookback absorbed into Model page focus chart |
| Quality Health | Deleted; absorbed into Triage |
| Scorecard | Deleted; absorbed into Triage |
| Smoothness | Deleted as standalone page; Smoothness logic absorbed into Model page's Smoothness tab |
| Specs | Deleted as standalone page; per-model spec management absorbed into Settings section |
| Settings | Replaced by V6 Settings with restructured sections |

### Architecture

```
V6App (CTk root)
├── ThemeManager — theme tokens, applied globally
├── Sidebar (left, fixed ~160px) — 4 items, accent-stripe active state
│                                   App-driven (sidebar.set_active called by app)
└── PageContainer (right, flex) — stacked CTkFrame, one per page
    └── PageBase subclass per page
        ├── _PageHeader (~40px) — page_title + header_actions() widgets
        └── _content (flex) — subclass owns this region entirely
```

All four pages constructed at app start. Page switching is `tkraise()` + lifecycle hooks (`on_show`/`on_hide`). State on each Page Frame persists across switches.

### Standard inheritance contract — `PageBase`

```python
class PageBase(ctk.CTkFrame):
    page_title: str = "Untitled"   # subclass overrides

    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color=theme.SURFACE, **kwargs)
        self.theme = theme
        self._build_chrome()
        self.build_content(self._content)

    def build_content(self, parent: ctk.CTkFrame) -> None:
        raise NotImplementedError

    def header_actions(self) -> List[ctk.CTkBaseClass]:
        return []   # optional override

    def on_show(self) -> None: pass
    def on_hide(self) -> None: pass
```

`V6App.show_page(name)` calls `on_hide` on the current page, `tkraise()` + `on_show()` on the new page, and `sidebar.set_active(name)`.

### State refresh strategy (all pages)

- `on_show()` is the refresh point. Each page reloads any DB-derived state on show.
- No background polling. Batch flow → no streaming need.
- Triage refresh is cheap (reads `model_metric_state` directly via `get_drifting_models`).
- Model page refresh is cheap (reads one model's state).
- Settings page does NOT refresh on show — its data is config + spec rows, slow-changing.
- Process page does NOT refresh; its state is the file-scan progress.

---

## Sub-spec 3a — App shell + sidebar nav + theme + PageBase

### Scope

Build the runnable V6 shell with four **placeholder** pages. Each placeholder shows a centered label like "Triage — coming in Spec 3b." This proves the shell + nav + theme works end-to-end.

### Files created

```
src/laser_trim_analyzer/gui/v6/
├── __init__.py
├── theme.py          — ThemeManager class with the constants above
├── sidebar.py        — Sidebar widget + internal _SidebarRow
├── page_container.py — PageContainer stacked-frame
├── page_base.py      — PageBase + _PageHeader
└── app.py            — V6App root + 4 placeholder page subclasses
```

### Files modified

- `src/laser_trim_analyzer/__main__.py` — add `--v6` CLI flag handling.

### Test file

- `tests/test_spec3a_shell.py` — unit tests for theme constants, sidebar active-state, page lifecycle hooks. NO mainloop integration tests (Tk in CI is flaky).

### Out of scope for 3a

- Real page content (3b/3c/3d/3e)
- Loading any DB data (3b/3c)
- Settings sliders (3d)
- Process file scanning (3e)
- Deletion of old GUI code (3e)

---

## Sub-spec 3b — Triage page

### Scope

The mission-critical landing page. Two zones: flagged-cards on top, scrollable browse list below.

### Layout

```
TriagePage extends PageBase
├── header_actions: [Refresh button]   (manual override on top of on_show refresh)
└── content:
    ├── FlaggedCardsZone (top)
    │   ├── Section label: "Needs attention (N)"
    │   └── Card grid (3-5 cards wide; wraps if needed)
    │       Each ModelAlertCard shows:
    │       - Model name (large, SIZE_TITLE)
    │       - Worst metric name (SIZE_BODY, TEXT_SECONDARY)
    │       - Magnitude (large, SIZE_DISPLAY, tier color)
    │       - Alert type badge ("Step change" / "Slow drift")
    │       - Tier-colored background (TIER_*_BG)
    │       - Click → navigate to Model page with this model + worst metric preselected
    │   Empty state when nothing flagged:
    │       "All models within tolerance — last processed {date}"
    │       in TEXT_SECONDARY
    │
    └── BrowseZone (bottom, flex)
        ├── Section label: "All models"
        ├── Search input — filters list as you type
        └── Scrollable list:
            Each row: model name + tier color dot + last-processed date
            Click → navigate to Model page with this model
```

### Data sources

- `get_drifting_models(db, sensitivity_preset)` for flagged cards. Returns `List[ModelAlertSummary]`, already sorted worst-first.
- Distinct-models query (or `get_drifting_models` returning all models with `tier=Stable` rows kept) for the browse list. Add a helper `list_known_models(db) -> List[ModelSummary]` in `ml/manager.py` if needed.

### Refresh

`on_show()` reloads both zones via threaded query, updates UI on main thread via `self.after(0, ...)`. Loading state shown if query > 200ms.

### Navigation flow

Click on flagged card → `V6App.show_page("model")` and the Model page receives a routing hint via `V6App.set_model_route(model, focus_metric)`. Model page's `on_show` reads the hint and configures focus accordingly. Hint is cleared after consumption.

### Out of scope for 3b

- The Model page itself (3c)
- Smoothness data in Triage (always-stale until you actively process smoothness files; uses same drift detector path so technically already supported)

---

## Sub-spec 3c — Model page

### Scope

Per-model investigation view. Focus chart + 8-pill glance row + 3 tabs (Drift Metrics / Smoothness / Units) + optional Predictor panel.

### Layout

```
ModelPage extends PageBase
├── header_actions:
│   ├── Model selector (search-as-you-type dropdown)
│   ├── Time window dropdown (30d / 90d / 365d / All)
│   └── "Export evidence pack" button
│
└── content:
    ├── PillRow (8 metric pills, horizontally arranged)
    │   Each pill: metric name + one-line summary + tier color
    │   Selected pill: ACCENT background; others: SURFACE
    │   Click → swap focus chart
    │
    ├── FocusChart (~280px tall)
    │   matplotlib time-series chart (uses existing gui/widgets/chart.py)
    │   X axis: file_date within selected window
    │   Y axis: metric value
    │   Overlays: baseline_mean line, baseline ± 2σ band,
    │             recent-batch highlight
    │   Header: metric name + alert-type badge + magnitude
    │
    └── TabRow (3 tabs)
        ├── "Drift Metrics" (default) — table of all 8 metrics
        │   columns: metric / tier / alert type / baseline mean ± std / recent mean / Δσ
        │   row selection mirrors PillRow selection
        ├── "Smoothness" — per-model smoothness summary
        │   max_smoothness_value trend chart, recent smoothness test list
        │   per-unit smoothness chart drill-down
        └── "Units" — sortable table of recent units
            columns: serial / file_date / overall_status / sigma_gradient / linearity_error
            click row → existing per-unit pre/post-trim chart in modal
            "Export to Excel" button → existing Excel export with this model's units
```

### Time window control

Header dropdown lets the user pick the focus chart's time window. Selection persists across page switches.

### Predictor panel (demoted)

Collapsible section below the tabs. Default collapsed. Reads existing per-unit predictor output. Marked "diagnostic — not part of daily flow."

### Data sources

- `get_model_drift_status(db, model)` → drift metrics tab + pill row
- Direct `TrackResult` query joined to `AnalysisResult` for focus chart series and Units tab
- `SmoothnessResult` query for Smoothness tab
- Existing per-unit chart logic (currently in V5 `analyze.py`) ports into a `UnitChartModal` reused by both Units tab and Smoothness tab

### Refresh

`on_show()` reads the current routing hint (set by Triage navigation or by user typing in model selector), reloads all sources for that model. Refresh cost: ~1 DB query per metric + 1 query for units list.

### Out of scope for 3c

- Real-time updates as files process (would require explicit refresh action)
- Cross-model views ("compare model X to Y") — anchor doc says per-model mission
- Editing exclude_points from this page (lives in Settings → Per-model specs section)

---

## Sub-spec 3d — Settings restructure

### Scope

Single Settings page with 5 collapsible sections. Includes the new sensitivity-preset slider, folded-in spec management, ML training controls, pricing config, DB cleanup. Wires in Spec 2's deferred first-startup auto-train hook.

### Layout

```
SettingsPage extends PageBase
└── content: scrollable vertical list of 5 SettingsCard widgets

    Section 1: Alert thresholds  (expanded by default)
    ├── Sensitivity preset slider (4 discrete stops):
    │       loose | standard | tight | strict
    ├── Live preview readout (updates 200ms after slider stops moving):
    │       "Would flag: X Warning, Y Drift, Z Out-of-Control"
    │   Powered by preview_alert_count(db, preset). Cheap.
    ├── Per-metric description list (educational):
    │       Each of the 8 watched metrics with one-line "what it watches"
    └── "Save preset" button — writes config.yaml and recomputes thresholds
        in model_metric_state in place (no full retrain).

    Section 2: Per-model specs  (collapsed by default)
    ├── Model selector dropdown
    ├── Linearity_spec override input
    ├── exclude_points list editor (matches existing V5 Specs page UX)
    └── "Save" button per model

    Section 3: ML training  (collapsed by default)
    ├── "Retrain drift detector" button (Spec 2 train_drift_detector())
    ├── "Retrain per-model thresholds" (existing threshold optimizer)
    ├── "Retrain predictor" (existing predictor)
    ├── "Retrain profiler" (existing profiler)
    └── Last training timestamps for each.

    Section 4: Pricing  (collapsed by default)
    └── Existing V5 pricing UI ported.

    Section 5: Database cleanup  (collapsed by default)
    └── Existing V5 cleanup UI ported.
```

### First-startup auto-train hook (Spec 2 deferred work)

In `V6App.__init__` (Spec 3a) or `DatabaseManager` startup (cleaner — at the data layer), after migrations run, check if `model_metric_state` is empty. If yes:

1. Show a one-time modal: "First-run drift training — this will take ~30 seconds for {N models}."
2. Run `train_drift_detector(db, preset)` in a background thread.
3. Progress callback updates the modal.
4. On completion, close modal. User is on Triage with real data.

If user closes the modal early, training continues; Triage shows partial results.

### Sensitivity preset preview implementation

```python
def _on_slider_change(self, new_preset: str):
    # Debounce: cancel any pending update, schedule new one for 200ms
    if self._pending_update:
        self.after_cancel(self._pending_update)
    self._pending_update = self.after(200, self._refresh_preview, new_preset)

def _refresh_preview(self, preset: str):
    # Run in background thread; update label on main thread
    def task():
        counts = preview_alert_count(self.db, preset)
        self.after(0, self._update_preview_label, counts)
    threading.Thread(target=task, daemon=True).start()
```

### Save behavior

"Save preset" persists `ml.drift_sensitivity = "<preset>"` to `~/.laser_trim_analyzer/config.yaml`, then recomputes thresholds in place for every `model_metric_state` row (no historical re-scan; uses the cached baseline). Runtime state (cusum/ewma) is preserved.

### Out of scope for 3d

- Per-model sensitivity overrides
- Bulk model deletion (lives in DB cleanup section already)
- Theme toggle (single dark theme for v1)

---

## Sub-spec 3e — Process page rewrite + retirement

### Scope

Rewrite the Process page using V6 patterns. Then delete all 8 old V5 pages, promote `gui/v6/*` up to `gui/`, delete the `--v6` flag.

### Process page rewrite

Full rewrite around V6 widget primitives. NOT a verbatim port.

```
ProcessPage extends PageBase
├── header_actions: [Cancel button — only visible while running]
└── content:
    ├── FolderPicker — browse + display selected folder path
    ├── Incremental mode toggle (default on)
    ├── "Start processing" button (disabled when no folder selected)
    │
    └── Progress section (visible after start):
        ├── ProgressBar
        ├── Status text ("Processing 47 / 1240: 8340-1-sn123_...xls")
        ├── Per-file result tally:
        │       Successes / Skipped (already processed) / Failures
        ├── Failure list (collapsible, last 10 failures with reason)
        └── On completion:
            "Processing complete. Triage updated."
            Button: "Go to Triage" → V6App.show_page("triage")
```

### Logic carry-over

Existing scan and batch logic from V5 `process.py` provides the algorithmic shell — copy the `processor` calls and threading patterns. UI widgets are all V6 primitives. No carryover of V5's custom CTk widgets — rebuild against V6 theme.

### Retirement commit (final commit of Spec 3e)

Single atomic commit that:

1. Deletes `src/laser_trim_analyzer/gui/app.py` (V5 root)
2. Deletes `src/laser_trim_analyzer/gui/pages/*.py` (all 8 old pages: dashboard, analyze, compare, trends, quality_health, scorecard, smoothness, specs, process — plus settings)
3. Deletes `src/laser_trim_analyzer/gui/widgets/scrollable_combobox.py` (V5-only)
4. Moves `src/laser_trim_analyzer/gui/v6/*.py` up to `src/laser_trim_analyzer/gui/*.py`
5. Moves `src/laser_trim_analyzer/gui/v6/widgets/*.py` into `src/laser_trim_analyzer/gui/widgets/`
6. Updates `__main__.py`: removes `--v6` flag handling, V6 is the only UI now
7. Updates `gui/__init__.py` to re-export from the new locations
8. Updates `CLAUDE.md` page count from "10 pages" to "4 pages" + page name list
9. Runs full regression sweep to confirm nothing broke

### Out of scope for 3e

- Adding new Process features (only rewriting existing ones)
- Theme switching
- Any UI changes to other pages (their final versions land in 3b/3c/3d)

---

## Cross-spec test infrastructure

### Test patterns

Per-sub-spec test files:
- `tests/test_spec3a_shell.py`
- `tests/test_spec3b_triage.py`
- `tests/test_spec3c_model.py`
- `tests/test_spec3d_settings.py`
- `tests/test_spec3e_process.py`

Each file follows the established pattern: import-then-call against headless CTk where feasible; mock `db.session()` returns where the test focus is widget behavior; integration smoke test (one per file) that constructs the page with a real `tmp_path` SQLite DB and walks the on_show path.

### Headless customtkinter caveats

CTk requires a root window for widget construction. Tests use `pytest-tkinter` style helpers or a session-scoped fixture that creates and destroys a `CTk()` root. No `mainloop()` calls in tests.

### Regression sweep after each sub-spec

After 3a, 3b, 3c, 3d, 3e individually:

```
pytest tests/test_spec1_untrimmed_sigma.py \
       tests/test_log_derived_bugfixes_2026_05_30.py \
       tests/test_5_8_2026_bugfixes.py \
       tests/test_spec2_multi_metric_drift.py \
       tests/test_spec3a_shell.py \
       tests/test_spec3b_triage.py \
       ... (sub-specs landed so far)
```

Must remain 0-fail throughout.

---

## Non-goals (Spec 3 entire)

- Mobile / tablet layouts — manufacturing PC monitor target.
- Auth, multi-user, role permissions — single-operator workstation.
- Web frontend — desktop only.
- Real-time streaming — batch flow.
- Cross-model views — per-model mission.
- Theme toggle — single dark theme.
- Animations / transitions — direct state changes.
- Configurable layouts — fixed sidebar position, fixed PageHeader, fixed PillRow / FocusChart proportions on Model page.
- Internationalization — English only.
- Drag-and-drop file ingest — folder picker only (matches V5 behavior).
