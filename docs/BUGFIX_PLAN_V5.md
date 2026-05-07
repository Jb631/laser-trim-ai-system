# Laser Trim Analyzer v5 — Code Review Bugfix Plan

**Created:** 2026-05-06
**Source:** 6-agent code review swarm (core, database, ML, GUI, security, utils/export)
**Total Issues:** 20 (5 critical, 7 high, 8 medium)
**Approach:** 5 phases, each independently deployable and testable

---

## Phase 1: Data Integrity Foundation

**Issues Fixed:** #1, #5, #11, #12, #16, #19
**Risk Level:** Low — these are one-line fixes that tighten validation
**Estimated Effort:** 1 session

| # | Severity | Issue | File(s) | Fix |
|---|----------|-------|---------|-----|
| 1 | CRITICAL | SQLite foreign keys never enabled — all FK constraints unenforced | `database/manager.py` | Add `PRAGMA foreign_keys = ON` after WAL mode pragma |
| 5 | CRITICAL | `all([]) == True` — error-state tracks report PASS in exports | `export/excel.py` | Check for empty list before `all()`: `all(vals) if vals else False` |
| 11 | HIGH | Negative linearity_spec from inverted limits crashes Pydantic | `core/parser.py` | Wrap return in `abs()`: `return abs((avg_upper - avg_lower) / 2)` |
| 12 | HIGH | optimal_slope default mismatch: ORM=0.0, migration=1.0 | `database/manager.py` | Fix migration DDL to `FLOAT DEFAULT 0.0` |
| 16 | MEDIUM | `session.query(Model).get(id)` deprecated (7 occurrences) | `database/manager.py` | Replace with `session.get(Model, id)` |
| 19 | MEDIUM | `datetime.now()` vs `utc_now()` mixed in timestamp writes | `database/manager.py` | Use `utc_now()` for `timestamp` fields; keep `datetime.now()` for `file_date` cutoffs |

### What Could Break
- **#1:** If the database has orphaned rows (tracks pointing to deleted analyses), inserts/updates will fail with IntegrityError. **Mitigation:** Run an orphan check query before enabling the pragma. Clean up any orphans found.
- **#5:** Strictly tightens validation. No regressions possible.
- **#11:** Files with inverted limits that previously produced negative linearity_spec (causing Pydantic crash) will now produce correct positive values. Some tracks may flip from PASS to FAIL (correctly).
- **#12:** Only affects NEW databases or newly-added columns. Existing records with baked-in 1.0 default stay as-is until reprocessed.

### Verification
1. Open Python REPL, connect to DB, run `PRAGMA foreign_keys;` — must return `1`
2. Create a mock result with all tracks having `sigma_pass=None`. Export to Excel. Verify shows "FAIL" not "PASS"
3. Process a file. Check `linearity_spec` in DB — must be positive
4. Fresh DB: verify `optimal_slope` defaults to 0.0
5. Check logs for SQLAlchemy deprecation warnings — should be gone
6. Process a file, check `timestamp` in DB — should be UTC

### Dependencies
None. This is the foundation phase.

---

## Phase 2: Numpy 2.x Compatibility

**Issues Fixed:** #3, #4, #9
**Risk Level:** Low — same fix pattern already proven in parser.py
**Estimated Effort:** 1 session

| # | Severity | Issue | File(s) | Fix |
|---|----------|-------|---------|-----|
| 3 | CRITICAL | `np.issubdtype(type(val), np.number)` breaks FT parser on numpy 2.x | `core/final_test_parser.py` (5 locations) | Replace with `isinstance(val, (int, float, np.integer, np.floating))` |
| 4 | CRITICAL | `_is_valid()` misses `np.float64(nan)` on numpy 2.x | `core/analyzer.py` line 91 | Change to `isinstance(v, (float, np.floating)) and np.isnan(v)` |
| 9 | HIGH | `predict_with_confidence` single-class logic inverted | `ml/predictor.py` line 431 | Change to `val = 1.0 if self.classifier.classes_[0] == 1 else 0.0` |

### What Could Break
- **#3:** FT files that previously parsed with zero data points will now parse correctly. New FT records may appear in the database. No data loss.
- **#4:** Theory voltage interpolation will now correctly handle NaN values instead of treating them as valid. Could change slope optimization results for files with missing theory data — results become more correct.
- **#9:** Models trained on all-pass data will return 0.0 probability instead of 1.0. Risk categories may change from HIGH to LOW (correctly).

### Verification
1. In REPL: `is_numeric(np.float64(3.14))` → must return `True`
2. In REPL: `is_numeric(42)` → must return `True` (plain Python int)
3. Process a known FT file — verify tracks are parsed (no "no track data" log)
4. `fill_missing_theory_volts([0.0, np.float64(np.nan), 5.0])` → verify interpolation works
5. Train predictor on all-pass model → `predict_with_confidence(...)` returns ~0.0, not ~1.0

### Dependencies
None. Independent of Phase 1.

---

## Phase 3: Database Robustness

**Issues Fixed:** #6, #7, #17
**Risk Level:** Medium — #6 changes lookup behavior which affects batch processing
**Estimated Effort:** 1 session

| # | Severity | Issue | File(s) | Fix |
|---|----------|-------|---------|-----|
| 6 | HIGH | `save_batch` uses filename-only lookup; `save_analysis` uses filename+path | `database/manager.py` | Make `save_batch` use `(filename, file_path)` filter like `save_analysis` |
| 7 | HIGH | Migration probe failures leave session in error state — no rollback | `database/manager.py` | Add `session.rollback()` after each failed probe query before ALTER TABLE |
| 17 | MEDIUM | Config save is non-atomic — crash mid-write corrupts file | `config.py` | Write to temp file, then `os.replace()` (atomic on all platforms) |

### What Could Break
- **#6:** Files that were incorrectly matched by filename-only in batch processing will now create separate records instead of overwriting. This is correct but means some old records may appear as duplicates until cleaned up. **Test carefully with batch processing.**
- **#7:** No risk. Strictly more robust error handling.
- **#17:** No risk. Same behavior on success, safer on failure.

### Verification
1. Create two files with identical names in different folders. Batch process both. Verify `SELECT COUNT(*) FROM analysis_results WHERE filename='test.xls'` returns 2
2. Manually add a column that a migration tries to add. Restart app. Verify no errors and other migrations still complete
3. Save config settings. Verify config file is valid YAML. (Crash simulation is optional but recommended.)

### Dependencies
Phase 1 should be deployed first (foreign keys active).

---

## Phase 4: Thread Safety and GUI Stability

**Issues Fixed:** #8, #10, #13, #14, #20
**Risk Level:** Medium — thread changes can introduce deadlocks if not careful
**Estimated Effort:** 1-2 sessions

| # | Severity | Issue | File(s) | Fix |
|---|----------|-------|---------|-----|
| 8 | HIGH | ML manager shared dicts have no locking | `ml/manager.py` | Add `threading.RLock`, acquire in all dict access methods |
| 10 | HIGH | Double-click launches duplicate processing runs | `gui/pages/process.py` | Disable button immediately before thread starts |
| 13 | MEDIUM | Export/query operations freeze GUI (main thread blocking) | `gui/pages/analyze.py`, `dashboard.py`, `settings.py` | Move heavy work to background threads |
| 14 | MEDIUM | `bind_all`/`unbind_all` hijacks mousewheel globally | `gui/pages/analyze.py`, `trends.py` | Use widget-scoped `bind()` instead of `bind_all()` |
| 20 | MEDIUM | Compare page checkbox vars and selected IDs never cleared | `gui/pages/compare.py` | Clear `_selected_ids` on filter change / data refresh |

### What Could Break
- **#8:** Lock ordering must be DB lock → ML lock (never reverse) to avoid deadlock. Verify no code path acquires ML lock then DB lock.
- **#10:** Fast clicks are silently ignored. This is desired behavior.
- **#13:** Export errors need explicit handling in the callback — ensure error dialogs still appear via `self.after()`.
- **#14:** Test mousewheel in dropdowns on Windows — `<MouseWheel>` event behavior varies by platform.
- **#20:** User loses selection on filter change. Correct behavior but may surprise users.

### Verification
1. Start ML training. While training runs, process a file. Verify no `RuntimeError: dictionary changed` errors
2. Rapidly double-click Process button. Check logs — "Starting batch processing" should appear exactly once
3. Export 1000+ results. Verify GUI stays responsive (can switch tabs during export)
4. Open dropdown in Analyze page. Scroll elsewhere — verify the other area scrolls, not the dropdown
5. On Compare page: select 3 checkboxes, change model filter, verify selection resets to 0

### Dependencies
Phases 1-2 should be complete (data correctness before UI).

---

## Phase 5: ML Security and Export Quality

**Issues Fixed:** #2, #15, #18
**Risk Level:** High — #2 invalidates existing pickle files, #15 changes export format
**Estimated Effort:** 1 session

| # | Severity | Issue | File(s) | Fix |
|---|----------|-------|---------|-----|
| 2 | CRITICAL | Pickle deserialization — no mandatory integrity check | `ml/predictor.py` | Make `.hash` file mandatory. Switch to HMAC-SHA256 with machine-local key. Refuse to load if hash missing or invalid |
| 15 | MEDIUM | Numeric values exported as formatted strings — breaks Excel sorting | `export/excel.py` | Write raw floats, apply `cell.number_format` for display |
| 18 | MEDIUM | Profiler pass_rate deflated by NaN sigma_pass from FT records | `ml/profiler.py` | Filter NaN sigma_pass records before computing pass_rate |

### What Could Break
- **#2:** **All existing pickle files without `.hash` files will fail to load.** The app falls back to formula-based thresholds. Re-training via Settings → Train ML regenerates everything from DB data. This is a one-time cost on upgrade. **Must document in release notes.**
- **#15:** Downstream tools/macros that parse exported Excel as strings will get floats instead. Technically a breaking change but strictly correct. **Ask James if any downstream workflows consume the Excel programmatically.**
- **#18:** Pass rates increase for models with mixed trim/FT data. Correct but visible change — dashboards will show higher pass rates. Add log message explaining the calculation change.

### Verification
1. Delete all `.hash` files. Start app. Verify predictors fail to load (check logs). Train ML. Verify new `.hash` files created. Restart — verify predictors load successfully
2. Export results to Excel. Open in Excel. Click sigma_gradient cell — formula bar should show a number. `=SUM()` on the column should work. Display should show 6 decimal places
3. Check profiler pass_rate for a model with both trim and FT data. Verify FT records with NaN sigma don't drag down the rate

### Dependencies
Phase 2 (#9 predictor fix) must be done before Phase 5 (#2 pickle security) — otherwise re-training bakes in the inverted single-class bug.

---

## Implementation Checklist

```
Phase 1: Data Integrity Foundation
  [ ] Add PRAGMA foreign_keys = ON
  [ ] Run orphan check on existing database first
  [ ] Fix all([]) == True in excel.py
  [ ] Add abs() to linearity_spec calculation
  [ ] Fix optimal_slope migration default to 0.0
  [ ] Replace session.query().get() with session.get() (7 places)
  [ ] Fix timestamp writes to use utc_now()
  [ ] Verify all Phase 1 changes
  
Phase 2: Numpy 2.x Compatibility
  [x] Fix is_numeric() in final_test_parser.py (5 locations)
  [x] Fix _is_valid() in analyzer.py
  [x] Fix predict_with_confidence single-class logic
  [x] Verify all Phase 2 changes

Phase 3: Database Robustness
  [ ] Fix save_batch to use (filename, file_path) lookup
  [ ] Add session.rollback() after migration probe failures
  [ ] Make config save atomic (temp file + os.replace)
  [ ] Verify all Phase 3 changes

Phase 4: Thread Safety and GUI
  [ ] Add RLock to MLManager for shared dict access
  [ ] Fix double-click race in process.py
  [ ] Move blocking exports to background threads
  [ ] Replace bind_all with widget-scoped bind
  [ ] Clear _selected_ids on Compare page filter change
  [ ] Verify all Phase 4 changes

Phase 5: ML Security and Export Quality
  [ ] Make pickle hash mandatory
  [ ] Switch to HMAC-SHA256
  [ ] Fix Excel export: raw numbers + number_format
  [ ] Fix profiler pass_rate NaN handling
  [ ] Document breaking changes (pickle re-training required)
  [ ] Verify all Phase 5 changes
```

---

## Risk Matrix

| Phase | Data Corruption Risk | Breaking Change Risk | Effort | Priority |
|-------|---------------------|---------------------|--------|----------|
| 1 | **Fixes corruption** | None | Low | Do first |
| 2 | **Fixes corruption** | None | Low | Do second |
| 3 | Low | Low | Medium | Do third |
| 4 | None | Low | Medium | Do fourth |
| 5 | None | **Medium** (pickle reload, Excel format) | Medium | Do last |

---

## Files Modified (Complete List)

| File | Phases | Changes |
|------|--------|---------|
| `database/manager.py` | 1, 3 | FK pragma, migration rollbacks, save_batch lookup, .get() deprecation, timestamps |
| `export/excel.py` | 1, 5 | all([]) fix, numeric formatting |
| `core/parser.py` | 1 | abs() on linearity_spec |
| `core/final_test_parser.py` | 2 | is_numeric() numpy 2.x fix (5 locations) |
| `core/analyzer.py` | 2 | _is_valid() numpy 2.x fix |
| `ml/predictor.py` | 2, 5 | predict_with_confidence fix, HMAC hash mandatory |
| `ml/manager.py` | 4 | Thread locking on shared dicts |
| `ml/profiler.py` | 5 | pass_rate NaN handling |
| `config.py` | 3 | Atomic save |
| `gui/pages/process.py` | 4 | Double-click guard |
| `gui/pages/analyze.py` | 4 | Background exports, mousewheel fix |
| `gui/pages/dashboard.py` | 4 | Background exports |
| `gui/pages/settings.py` | 4 | Background DB operations |
| `gui/pages/trends.py` | 4 | Mousewheel fix |
| `gui/pages/compare.py` | 4 | Clear stale selections |
