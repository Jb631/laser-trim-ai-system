# Whole-App Audit — Consolidated Fix Plan

**Date:** 2026-06-01
**Source:** 6-agent whole-app correctness/QA audit (analyzer, parsers, processor, database, ML, GUI/export),
findings adversarially verified by reading the code. Plus the **sigma-signal correction** raised by James.
**Status:** Approved direction. Execute in batches, TDD, lowest-risk first.

## Decisions (locked)

- **D-SIGMA — Untrimmed sigma drives drift; post-trim sigma is a quality gate only.**
  `sigma_gradient` (analyzer.py:213) is computed on **post-trim** (corrected) errors. That is a *lagging,
  trim-confounded* signal and must **not** be used to detect element-production drift. The proper signal is
  `untrimmed_sigma_gradient` (analyzer.py:225-247, added by Spec 1) computed on the raw sweep. Going forward:
  - **Drift detection** keys on the **upstream** signals: `untrimmed_sigma_gradient`, `untrimmed_resistance`,
    `measured_electrical_angle`, `trim_pass_count`, `resistance_change_percent` (process inputs) plus
    `linearity_error` and `max_smoothness_value` (the outcomes handed to engineers). **Post-trim
    `sigma_gradient` is removed from the drift watch list.**
  - **Post-trim `sigma_gradient` / `sigma_pass`** remains only as an **internal finished-unit quality
    indicator** (it is *not* a customer spec — zero-tolerance linearity is). It is not a drift signal.
- **D-BRANCH — Pure correctness bugfixes land on `main` (V5) and merge forward to `V6`. The drift-signal
  redesign + detector wiring/ARL (Batch I) lands on `V6`** (that's where the multi-metric detector lives).
- **D-REPROCESS — The sigma/analyzer changes require reprocessing historical files** (to populate
  `untrimmed_sigma_gradient` and recomputed metrics on old records) **and retraining** the drift detector.
  This is the normal reprocess workflow; it is a gated step after the analyzer/drift batches land.

## How to read this

Each batch is independently shippable and testable. Findings are referenced by their audit IDs
(C1, H1…, plus Medium/Low). Severity and evidence live in the audit report (this session's transcript);
this doc is the **execution plan**. Every batch: write failing tests first → implement → green → commit.
After each `main` batch, run the existing suite; after the relevant batches, reprocess + retrain.

---

## Batch A — UNTRIMMED-denominator family (main) — **start here**

**Why first:** highest-value, lowest-risk, self-contained; wrong yield numbers are shown to QA today and
screens disagree with each other.

**Findings:** H1 (DB yield/Cpk-rate denominators), H2 (`BatchSummary` missing UNTRIMMED bucket + GUI
mislabels it error), H7 (Trends page client-side recompute incl. UNTRIMMED + unweighted averaging).

**Changes:**
1. Add one shared helper for "gradeable (trimmed) population" and use it everywhere a yield denominator is
   built. Concretely, exclude `UNTRIMMED` (file-level `overall_status`, track-level `status`) from the
   denominators in: `get_overall_stats` (manager.py:2043, 2048-2049), `get_dashboard_stats` daily-trend
   (1744-1782), `get_system_comparison` (2052-2094), `get_trending_worse_models` (3946-4048),
   `get_model_stats` (2414-2476), `get_escape_overkill_analysis` (2159-2237 — add
   `linearity_pass.isnot(None)` / status filter), and the LOW spots (`get_linearity_margin_analysis`,
   `get_anomaly_rate_by_model`). Copy the correct pattern already in `get_active_models_summary`.
2. `BatchSummary`: add `untrimmed: int = 0`; add the `elif AnalysisStatus.UNTRIMMED` branch in
   `processor.py:_update_summary` (1071-1085); set `pass_rate = passed / (processed - untrimmed)`. Fix the
   GUI `else` in `gui/pages/process.py` so UNTRIMMED is not counted as an error.
3. `trends.py:2276-2278` — filter `sigma_pass is not None` before counting (or consume the server
   `sigma_pass_rate`). `trends.py:2086-2095` — volume-weight the Active-Models tiles by each row's `total`
   (`Σ rate·n / Σ n`), or relabel them "avg of per-model rates."

**Tests:** seed DBs with a known PASS/WARNING/FAIL/UNTRIMMED mix; assert each rate uses the trimmed
denominator and that headline == trend == per-model for the same window. Regression: existing yield tests.

**Acceptance:** every yield/pass-rate on every screen excludes UNTRIMMED from the denominator and the
three views agree for the same model/window.

---

## Batch B — One-sided linearity Cpk/Ppk (main)

**Findings:** H3.

**Changes:** linearity error is a non-negative max-abs deviation → a **one-sided** characteristic. In
`core/cpk.py` (or a linearity-specific wrapper), set `LSL = 0`/`None`, report `Cpu/Ppu` and a one-sided
`Cp = (USL − target)/(3σ)` form; stop emitting the meaningless `cpl`/`−spec` term and the 2×-inflated
two-sided `Cp/Pp`. Mirror the already-correct one-sided treatment in `get_model_cpk` (sigma). Keep the
signed/bilateral path available for any genuinely two-sided variable.

**Tests:** feed a half-normal-ish positive error set; assert `Cpu == (USL−mean)/(3σ)` and that `Cp` is the
one-sided value (not 2×). Snapshot the Scorecard/Trends Cpk numbers before/after for sanity.

**Acceptance:** linearity Cp/Pp/Cpk are one-sided and consistent with the sigma Cpk convention.

---

## Batch C — Stop fabricating / mis-passing measurements (main)

**Findings:** H6 (FT shop-test fabricates `measured=0.0`/`error=0.0`), C2 (generic smoothness uses signed
`max()` and can pass a failing unit).

**Changes:**
1. `final_test_parser.py:1156-1173` — on a blank measured/error cell, **skip the row** (do not append
   position/theory/limits for it), exactly like the Format-1/2 paths. No 0.0 substitution.
2. `smoothness_parser.py:285, 333` — `max_smoothness = max(abs(v) for v in vals)`,
   `avg_smoothness = mean(abs(v))`, matching the Betatronix path, so deviation magnitude drives pass/fail.

**Tests:** FT shop-test file with a blank measured cell → arrays stay aligned and no phantom error;
generic smoothness file whose worst excursion is negative and exceeds spec → correctly **fails**.

**Acceptance:** no fabricated measurements; signed-vs-magnitude smoothness cannot pass a failing unit.

---

## Batch D — Incremental dedup correctness (main)

**Findings:** H4 (skip-by-path not hash), H5 (sticky `success=True` skip).
**Prerequisite:** confirm with James whether trim stations **reuse filenames** (e.g. fixed `test.xls`
export). If yes, H4 is data-loss-class; if every export is uniquely named, it's lower risk.

**Changes:**
1. `processor.py` `_load_processed_hashes`/`_is_processed` (1139-1160, 1205-1220) — make the in-memory skip
   set **hash-based** (load `file_hash`, compare `calculate_file_hash(path)` — already mtime-cached), or
   require a hash confirmation before a path-hit skip. Rename the method to match reality.
2. `_mark_file_skipped` / `_record_processed_file` — mark skipped/non-trim rows `success=False` (or a
   `skipped` flag), or have `_record_processed_file` **update** an existing-hash row instead of no-oping, so
   a misclassified file can be reprocessed.

**Tests:** same path, new content → reprocessed (not skipped); same content, new path → not duplicated;
a file first skipped then legitimately reprocessed as trim links a `ProcessedFile` row.

**Acceptance:** dedup is content-identity based; a transient misclassification is not permanent.

---

## Batch E — ML honesty: kill leakage, validate properly (main)

**Findings:** C1 (predictor target leakage), H10 (threshold optimizer trains sigma against a linearity
label, no holdout), H11 (random split leaks repeated serials).

**Changes:**
1. **Predictor** (`predictor.py`, `manager.py:307,904-959`, root `analyzer.py:545`): the label
   `passed = (fail_points == 0)` is computed from features `fail_points`/`linearity_error`/`error_to_spec`.
   Either (a) repurpose to predict **final-test** outcome from **trim** features, dropping all
   linearity-derived features; or (b) honestly demote/remove it (aligns with the mission's predictor
   demotion) and stop surfacing its inflated accuracy/AUC. Recommended: (b) for now — relabel the panel
   "diagnostic, not validated" and remove the misleading metrics; revisit (a) as a separate effort.
2. **Threshold optimizer** (`threshold_optimizer.py`): report **out-of-sample** confusion-derived
   confidence (held-out split), and document which outcome it targets. Given D-SIGMA, retarget it to the
   **untrimmed** sigma if it's meant to gate process health.
3. **Splits** (`predictor.py:217-296`): carry `serial`; use `StratifiedGroupKFold`/`GroupShuffleSplit`
   keyed on serial (repeated trims are valid), or a temporal split.

**Tests:** assert no linearity-derived feature is in the predictor's feature set under option (a)/(b);
assert grouped split keeps a serial's rows on one side.

**Acceptance:** no metric is reported from a leaked model; CV respects repeated-serial grouping.

---

## Batch F — Analyzer robustness & derived metrics (main)

**Findings (Medium):** sigma inflation from tiny `dx` (analyzer.py:411-416); `max_deviation_position`/
`deviation_uniformity` ignore `exclude_indices` and aren't NaN-safe (298-312); FT Format-1 position
auto-detect can pick the wrong column (final_test_parser.py:485-501); parallel vs sequential hard-error
divergence (processor.py:797-799 vs the sequential loop).

**Changes:** reject gradient samples where `dx < 0.1·median_dx`; make the `max_deviation*`/uniformity block
exclusion-aware + NaN-filtered; validate the FT position column's physical range and flag a data-quality
warning on fallback; route both processing paths through one shared result handler so a hard exception
yields `processed += 1` + `errors += 1` in **both**.

**Acceptance:** sigma not dominated by near-coincident points; reported "worst deviation" respects
exclusions; parallel and sequential summaries match on hard errors.

---

## Batch G — UI presentation correctness (main)

**Findings:** sigma scatter y-axis clip hides worst fails (chart.py:1452-1454, also `plot_drift_chart`,
`plot_histogram`); dashboard `warnings=0` fallback (dashboard.py:677-681); smoothness badge binary
(smoothness.py:216,222).

**Changes:** when a clipped outlier is a Fail/OOC point, raise `y_max` to include it or draw a clamped
"↑ off-scale" fail marker; pull `warnings` from `stats` (don't hardcode 0); add a neutral badge color for
non-PASS/non-FAIL smoothness statuses.

**Acceptance:** no failing point is hidden by axis clipping; counts never silently show a false `⚠0`.

---

## Batch H — Low-severity cleanups (main)

Memory-throttle constants vs comments (processor.py:51-53); extend the `'Pass'→'PASS'` migration to FT/
smoothness tables (manager.py:389-397); `get_ml_staleness` naive/aware datetime (manager.py:4423); add an
index on `analysis_results.unit_id`; note `%W` week-bucket labels. Batch together, low risk.

---

## Batch I — Drift-signal redesign + detector wiring (V6)

**This is the heart of D-SIGMA + the audit's H8/H9. Lands on `V6`.**

**Findings:** D-SIGMA (drift uses untrimmed, drop post-trim sigma), H8 (detector never advanced on live data;
baseline over all history; `baseline_cutoff_date` unused → always "Stable"), H9 (per-observation FP instead
of ARL; no multiplicity correction → "flags everything" once wired).

**Changes:**
1. **Watched metrics** (`ml/drift_types.py` `WATCHED_METRICS`): remove `sigma_gradient` (post-trim); the
   upstream/outcome set becomes `untrimmed_sigma_gradient`, `untrimmed_resistance`, `linearity_error`,
   `measured_electrical_angle`, `trim_pass_count`, `resistance_change_percent`, `max_smoothness_value`
   (7 metrics). Update the Spec 3 plans/foundations: the "8-pill glance" becomes 7 (or shows post-trim
   sigma as a separate, non-flagging *quality* pill — implementer's call, documented).
2. **Fixed baseline + live advance** (`drift_training.py`, `multi_metric_drift_detector.py`, ingest hook):
   compute the baseline from samples **up to `baseline_cutoff_date`** (a fixed in-control reference), then
   **replay** post-cutoff samples through `MetricDetector.update()` and **persist** `cusum_pos/neg`,
   `ewma_state`, `last_processed` back to `model_metric_state`. Add an "advance on new data" step to the
   processing flow (or a recompute) so the detector reflects current state instead of always reading 0.
3. **ARL-based thresholds + multiplicity** (`drift_types.py` presets, `multi_metric_drift_detector.py`
   `compute_thresholds`): define presets in **ARL₀** terms (target ~200–500) and solve `h`/`L` against ARL
   (CUSUM via Siegmund; EWMA via Crowder/Lucas-Saccucci), allocating the budget across the 7 metrics
   (`p_metric = p_family/7`, or per-metric ARL = 7× the model target). Add a regression test that feeds a
   long in-control noisy stream and asserts the empirical alarm rate matches the design ARL₀.
4. **Legacy V5 detector** (`ml/drift_detector.py`, running on `main`): one small main-branch bugfix — point
   it at `untrimmed_sigma_gradient` instead of post-trim `sigma_gradient` so the deployed app monitors the
   correct signal until V6 graduates. (Optional if you'd rather rely solely on V6; recommended to do it.)

**Tests:** ARL₀ regression (in-control stream stays Stable to target rate); a synthetic drifting-untrimmed-
sigma stream flags WARNING→DRIFT→OOC at the right magnitudes; `get_drifting_models` returns the drifted
model after an advance step (proving H8 fixed).

**Acceptance:** drift is detected on the **untrimmed** signal, the detector reflects live state, and an
in-control process does **not** "flag everything."

---

## Batch J — Reprocess + retrain (gated, after F & I)

After the analyzer (Batch F) and drift (Batch I) changes land: reprocess all files (repopulate
`untrimmed_sigma_gradient` and recomputed metrics on historical records), then retrain the drift detector
so baselines are built from the corrected signals. Verify Triage/Model show real, correctly-signalled
drift. Update `docs/` session notes.

---

## Sequence & dependencies

```
main:  A → B → C → D → E → F → G → H   (each: tests → impl → green → commit; merge-forward to V6 periodically)
V6:    (after merge-forward of F) → I  (drift redesign) 
both:  J  (reprocess + retrain)  — gated on F (analyzer) and I (drift)
```

- A–H are independent; do A first (safest/highest-value), then by severity.
- I depends on F being merged forward (it consumes the corrected analyzer signals) and updates the Spec 3
  plans' metric list.
- J is last and is operational (reprocess + retrain), gated on F and I.

## Acceptance for the whole effort
1. Every yield/Cpk number shown to QA uses the gradeable denominator and is consistent across screens.
2. No fabricated/mis-passed measurements; no failing point hidden by a chart.
3. No ML metric reported from a leaked model.
4. Drift is detected on the **untrimmed** process signal, reflects live data, and does not false-alarm on
   an in-control process.
5. Full `pytest tests/` green throughout; reprocess + retrain completed and verified.
</content>

---

## Completion status (2026-06-01)

All code batches implemented, tested, and committed. Running test suite green
throughout (148 on main; full V6 suite green incl. drift-advance proofs).

| Batch | Branch | Commit(s) | Status |
|---|---|---|---|
| A — UNTRIMMED denominators (H1,H2,H7) | main | 65be558, 40fe41e | ✅ done |
| B — one-sided linearity Cpk (H3) | main | 996fcf7 | ✅ done |
| C — FT/smoothness fabrication (H6,C2) | main | ea36b6d | ✅ done |
| D — content-hash dedup (H4,H5) | main | 4211e14 | ✅ done |
| E — ML leakage/validation (C1,H11) | main | 7553b5c | ✅ done (H10 → Batch I retarget) |
| F — analyzer robustness | main | 3de4c87 | ✅ done (FT pos auto-detect deferred) |
| G+H — UI + low-severity | main | d85bfc2 | ✅ done |
| I — drift signal redesign | main + V6 | 3c11ebd (main); 15d6412, 9477ed2 (V6) | ✅ core done |
| J — reprocess + retrain | operational | — | ⏳ runbook below (user-executed) |

**Batch I detail:**
- main `3c11ebd` — legacy V5 drift detector keys on untrimmed_sigma_gradient (D-SIGMA), both train + apply paths; post-trim sigma stays the quality gate. **Deployable now.**
- V6 `15d6412` — drop post-trim sigma from WATCHED_METRICS (7 metrics); Bonferroni FP budget across metrics (the big "flags everything" lever).
- V6 `9477ed2` — fixed-baseline + replay training and `advance_drift_state()` so the detector reflects drift in history and responds to new data (fixes the H8 "always Stable" bug). Tests prove a drifted model flags.

### Batch J — reprocess + retrain (operational runbook; run on the real DB)

Required because D-SIGMA + the analyzer changes change stored signals/baselines.

1. **Back up** `./data/analysis.db`.
2. **Reprocess all files** (Process page, incremental OFF, or re-point at the source folders) so every record gets `untrimmed_sigma_gradient` + the recomputed analyzer metrics. (Reprocessing is the documented normal mechanism.)
3. **Retrain** in Settings: drift detector (now on untrimmed sigma) + per-model threshold optimizer/predictor/profiler.
4. **Verify**: per-model pages show drift on the untrimmed signal; yield numbers agree across Dashboard/Trends; Cpk reads one-sided.

### Remaining follow-ups (documented, not blocking)

- **H9 — full ARL₀ threshold calibration** (V6): the per-observation FP target + Bonferroni is implemented; the deeper fix is to derive `h`/`L` from run-length (ARL₀) targets (CUSUM via Siegmund; EWMA via Crowder/Lucas-Saccucci) so a single metric doesn't false-alarm every ~20 parts. Captured in the Spec 3 foundations as H9.
- **Wire `advance_drift_state()` into the Process flow** (Spec 3e) so Triage reflects new data automatically.
- **Merge `main` → `V6`** to bring audit Batches A–H forward (per D-BRANCH; manager.py will need careful conflict resolution since both branches edited it).
- **FT Format-1 position-column auto-detect** hardening (Batch F medium, deferred — heuristic).
