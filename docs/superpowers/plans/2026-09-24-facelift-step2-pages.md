# Facelift Step 2 — The Other Six Pages — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relayout Investigate (the model page), Home, Settings, Dashboard, Triage and Process on step 1's building blocks so the most important thing on each page comes first, in words.

**Architecture:** Two small shared additions (a wrap-to-width helper in `blocks.py`; `rows_per_group` and `groups` options on `FindingsView`), then one task per page, each changing only that page's module and its widgets, each proven by tests plus a clean `render_pages.py --audit` at both window sizes.

**Tech Stack:** CustomTkinter 5.2.2, matplotlib (FocusChart), SQLAlchemy reads through `app.db`.

**Spec:** `docs/superpowers/specs/2026-09-24-facelift-step2-pages-design.md` (rulings O and 1–6). Step 1's spec (`2026-09-23-design-system-and-findings-page-design.md`) still binds the look.

## Global Constraints

- **Never open `data/analysis.db` read-write.** Since 2026-09-24 `Config.load()` on the Mac resolves to it, so a bare `get_database()` in an ad-hoc script now reaches production: inject a tmp DatabaseManager into BOTH `laser_trim_analyzer.database.manager._db_manager` and `laser_trim_analyzer.database._db_manager`. Real-data runs use a COPY (`cp data/analysis.db /tmp/qa_copy.db`), deleted afterwards.
- `tk_root` stays function-scoped; app tests use `make_app`; never both in one test. Under `make_app` the window is withdrawn, so `winfo_ismapped()` is ALWAYS false — assert packing with `winfo_manager()`.
- No hex colour in `gui/v6` outside `theme.py`. Sentence case, verdict badges excepted. At most ONE teal-filled (`primary_button`) per screen. Numbers in `theme.mono`. Lasers via `core.models.laser_label()`.
- A failure is NAMED in a banner (`blocks.banner`, check tone) — never rendered as "no data" or zeros.
- No fixed pixel wraplength on page-width text; use `blocks.wrap_to_width` (Task 1).
- Workers never call Tk: loads run on a thread and post back through the page's existing `safe_after`.
- `strftime("%-d")` is forbidden; bind datetimes into `text()` as `f"{dt:%Y-%m-%d %H:%M:%S.%f}"`.
- Never `git stash` / `git checkout -- <file>` / `git reset`. Stage by explicit path; never `git add -A`; never stage `.claude/settings.local.json`. Commit trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- The gate: `.venv/bin/python scripts/run_test_gate.py` (read its summary line), plus `scripts/render_pages.py <copy> qa_output/pages --audit` clean at both sizes for the page you changed. Every new state/guard is mutation-checked.

## File structure

| file | change | responsibility |
|---|---|---|
| `gui/v6/widgets/blocks.py` | modify | `wrap_to_width(label, container, padding=0)` |
| `gui/v6/widgets/findings_view.py` | modify | `rows_per_group`, `groups` options |
| `gui/v6/pages/model_page.py` + its widgets | modify | Investigate relayout |
| `gui/v6/widgets/focus_chart.py` | modify | the legend no longer covers the off-scale note |
| `gui/v6/widgets/findings_tab.py` | modify | facts restyled with the blocks |
| `core/yield_stats.py` | modify | the raw-datetime bind at ~350 |
| `gui/v6/pages/home_page.py` | modify | Home relayout |
| `gui/v6/pages/settings_page.py`, `gui/v6/widgets/settings_card.py` | modify | order, titles |
| `gui/v6/pages/dashboard_page.py` + its panels | modify | banners per loader, layout |
| `gui/v6/pages/triage_page.py`, `process_page.py` | modify | rows, restyle |

---

### Task 1: Two shared additions

**Files:** Modify `gui/v6/widgets/blocks.py`, `gui/v6/widgets/findings_view.py`; tests in `tests/test_blocks.py`, `tests/test_findings_view.py`.

**Interfaces:**
- `wrap_to_width(label: ctk.CTkLabel, container, padding: int = 0) -> None` — binds `container`'s `<Configure>` (add="+") so `label`'s wraplength = container width − padding (never below 120); sets it once immediately.
- `FindingsView(master, theme, *, on_open=None, include_empty=True, rows_per_group: Optional[int] = None, groups: Optional[Sequence[str]] = None)` — `rows_per_group` defaults to `presentation.ROWS_PER_GROUP`; `groups` limits which group keys are drawn (None = all).

- [ ] **Step 1: Failing tests:** a label wrapped to a 400 px container reports wraplength 400 − padding after `update()`; resizing the container to 300 updates it; FindingsView with `rows_per_group=3` shows three rows and "Show all N"; `groups=("yield",)` draws only that group; the defaults are unchanged (existing tests stay green).
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** PASS; mutation-check both options; gate; commit `feat(ui): wrap text to its container, and a compact findings view`.

---

### Task 2: Investigate — what's worth changing comes first

**Files:** Modify `gui/v6/pages/model_page.py`; tests in `tests/test_spec3c_model.py` (and the model-page test files that pin its structure).

- [ ] **Step 1: Failing tests (make_app, a tmp DB with one model and a cached finding — reuse the existing model-page fixtures):**
  1. After loading a model, the page caption (the `PageBase` caption label's text) equals the drift-watch verdict sentence the old `_verdict` label showed.
  2. A "Worth changing on this model" group header precedes the pills and the stats table in packing order, and holds a `FindingsView(include_empty=False, on_open=None, rows_per_group=3, groups=("yield","laser_time","check"))` fed the model's cached findings.
  3. No findings → the one line "Nothing worth changing stands out for this model." is packed (`winfo_manager()`), the view is not.
  4. A failed findings load → a check-tone banner naming "findings"; never the "nothing worth changing" line.
  5. The spec-mismatch and load-failure notices are `blocks.banner`s (check tone), packed only when they have something to say.
  6. The σ key is ONE line under the pills ("σ = how far the last lot sits from this model's history of lots — a drift signal, not a spec"); the full explanation is in the Drift metrics tab.
- [ ] **Step 2:** FAIL. **Step 3: Implement** in `build_content` / the apply path; the old `_verdict` label is removed (its text goes to `set_caption`); keep every loader's independent guard. **Step 4:** PASS; mutation-check 3 and 4 (swap which is packed on failure → red); `render_pages.py --audit` on a copy for `model:*`; gate; commit `feat(investigate): what is worth changing comes first`.

---

### Task 3: Investigate — the chart, the tabs, the facts, the predictor

**Files:** Modify `gui/v6/widgets/focus_chart.py`, `gui/v6/pages/model_page.py` (tab names), `gui/v6/widgets/findings_tab.py` (facts), `gui/v6/widgets/predictor_panel.py`, `core/yield_stats.py`; tests accordingly.

- [ ] **Step 1: Failing tests:** (a) FocusChart with off-scale points and a legend: the legend's bounding box (`legend.get_window_extent()`) does not overlap the off-scale note's (`text.get_window_extent()`) — render with the Agg canvas in the test; (b) the seven tab names are exactly "Drift metrics", "Smoothness", "Units", "Final test units", "Trim vs final test", "History", "Findings", and the Findings-tab route (`consume_model_tab`) still selects Findings; (c) the Findings tab's fact sections use `blocks.group_header` (count = the number of rows each section lists) instead of bare labels; (d) the predictor panel's first line is "How well the final-test predictor would have called these units" and contains no pass/fail word for a unit; (e) `core/yield_stats.compute_trim_necessity` runs with `-W error::DeprecationWarning` (no raw datetime bound).
- [ ] **Step 2:** FAIL. **Step 3:** implement (legend `loc="upper left", bbox_to_anchor=(1.0, 1.0)` outside the axes with `fig.subplots_adjust(right=...)`, or the note moved above the plot — choose what the test and the audit accept at both sizes; update every test/constant that named the old Title Case tabs — 6 references). **Step 4:** PASS; mutation-checks; audit; gate; commit `feat(investigate): chart note clear of its legend, sentence-case tabs, facts on the blocks`.

---

### Task 4: Home — what needs attention after what's new

**Files:** Modify `gui/v6/pages/home_page.py`; tests in `tests/test_home_page.py` (or the existing home tests — find them).

- [ ] **Step 1: Failing tests (make_app, tmp DB with cached findings and one processed file):** the caption reads "Last processed {date} · {N} worth changing · {M} drifting now" (N = yield-group rows, M = focus entries; date via `f"{dt.day} {dt:%b}"`); a "Worth changing" section with `FindingsView(groups=("yield",), rows_per_group=3, include_empty=False, on_open=<routes to the model's Findings tab>)` and an "Open Findings" `link_button`; the focus list sits under the heading "Drifting now"; the two notices (legacy final tests, unreadable files) are quiet banners, packed only when non-zero; the folders line and summary wrap via `wrap_to_width`; still exactly one teal button ("Process everything new").
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** PASS; mutation-check the empty-findings state (a quiet line, never a blank gap); audit `home`; gate; commit `feat(home): what needs attention, after what's new`.

---

### Task 5: Settings — order and titles

**Files:** Modify `gui/v6/pages/settings_page.py` (and `widgets/settings_card.py` if its title styling needs the blocks); tests in the settings test file.

- [ ] **Step 1: Failing test:** the cards, in order, are titled "Ingest folders", "Backlog — active models and pricing", "Alert thresholds", "Per-model specs", "ML training", "Database"; only "Ingest folders" starts expanded; each card's header uses `SIZE_HEADING` text.
- [ ] **Step 2:** FAIL. **Step 3:** implement (the Ingest folders title keeps its hint about Home as a caption inside the card, not in the title). **Step 4:** PASS; audit `settings`; gate; commit `feat(settings): the setting Home needs first, titles in sentence case`.

---

### Task 6: Dashboard — every failed loader is named

**Files:** Modify `gui/v6/pages/dashboard_page.py` (+ `widgets/yield_panel.py`, `widgets/worst_models_list.py` for row styling); tests in the dashboard test file.

- [ ] **Step 1: Failing tests (make_app):** monkeypatch `compute_yield` to raise → a check banner naming "yield" and NO zero shown in either yield panel (today the except block renders zeros with no log); the same for the company trend ("company trend") and the cost priorities ("priorities"), each independently (one failing never blanks the others); the caption reads "Laser {x}% · final test {y}% over the last {window}" from the loaded yields (and "—" when a yield is unknown); the lowest-yield list draws `blocks.row`s (model mono, statement, readout mono).
- [ ] **Step 2:** FAIL. **Step 3:** implement — split `_query` so each loader returns (value, error) and `_apply` shows one banner listing the failed loaders; log every failure with `logger.exception`. **Step 4:** PASS; mutation-check (swallow one error again → its test red); audit `dashboard`; gate; commit `fix(dashboard): a failed query is named, never drawn as zero`.

---

### Task 7: Triage and Process

**Files:** Modify `gui/v6/pages/triage_page.py`, `gui/v6/pages/process_page.py` (+ `widgets/focus_list_zone.py` / `browse_zone.py` only for headings and wraps); tests in their test files.

- [ ] **Step 1: Failing tests:** Triage headings "Needs a look" and "All models"; captions wrap via `wrap_to_width`. Process: one teal "Start processing"; the incremental checkbox's `checkmark_color` is `TEXT_INVERSE`; after a run completes a `link_button` "See what changed" appears and routes to Findings; the db-info line wraps via `wrap_to_width`.
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** PASS; audit `triage`, `process`; gate; commit `feat(triage, process): clear headings, one button, a way to what changed`.

---

### Task 8: Close-out

- [ ] The whole gate; `render_pages.py --audit` on a copy — every page, every Model tab, both sizes, clean; `app_qa_sweep.py` and `chart_qa_render_all.py` on the copy (look at the focus charts' off-scale notes); delete the copy.
- [ ] `BRING_TO_WORK.md`: a page-by-page "what to look at" for the new layouts; `TRACKER.md`: F3 ticked with what each page now leads with.
- [ ] `python scripts/check_no_customer_values.py origin/main..HEAD` → 0 problems.
