# ML Capability Audit

**Date:** 2026-05-07
**Status:** Findings only — implementation deferred. Read this fresh before deciding what to tackle.

User question that drove this audit: "I want to make sure I'm getting the most out of machine learning."

## Executive summary

The ML system is architecturally complete (per-model predictor, threshold optimizer, drift detector, profiler — all 134 models trained) but its value is being left on the table in three areas. (1) The highest-value use case — pre-shipping batch screening that ranks units by predicted FT failure probability — does not exist anywhere in the GUI. `failure_probability` is computed and stored but never aggregated to a batch summary. (2) There is no accuracy feedback loop: the system never tells the user whether last month's predictions were correct, so the 134 trained predictors are a black box with no trust signal. (3) Drift detection signals are stored and computed correctly but respond to nothing automatically — the user must notice a drift badge, navigate to Settings, and click "Train Models" manually.

## Coverage map

The four ML components have different sample-size requirements (`ml/manager.py:99-103`):

- Profile: 10 samples minimum (always trains first)
- Threshold optimizer: 20 samples minimum
- Drift detector: 30 samples minimum (requires 21-sample baseline split from 70% of data)
- Predictor (RandomForest): 50 samples minimum

Per the DB:

- All 134 models with `model_ml_state.is_trained=True` have at least a profile (10+ samples).
- Models with <20 records silently get a formula fallback threshold (`spec / 200`) with `method='fallback'`, `confidence=0.3`. The user sees no indication of this in the GUI — they can't tell when ML is actually being used vs falling back.
- Models with <30 records have no drift baseline; they never appear in the Drift tab.
- Models with <50 records have no RandomForest predictor; no `ML XX%` badge in the Analyze list (just absent, no fallback text).

`db.get_ml_staleness()` flags models needing retrain at >50 new records since last training. Surfaced only in Settings, not on operational pages. No per-component breakdown — the message just says "Retrain needed: 12 models" without distinguishing "5 of those just qualified for predictor" vs "7 are drift-only."

## Surfacing gaps (where each ML output lives now vs where it could)

### `failure_probability` (RandomForest prediction)

**Currently:** ML probability badge on Analyze list rows when ≥50% (amber) / ≥75% (red). Stored in DB. Used during live processing for risk_category assignment.

**Where it should be but isn't:**

- **Process page completion summary.** After a 200-file batch, the user sees pass/warn/fail counts but not "38 units predicted high-risk for FT failure — review before assembly?" The data is computed — just never aggregated. *Answers: which units to hold this morning.*
- **Sortable column on Analyze list.** Can't sort by failure_probability or filter to "show only ML-flagged." *Answers: pre-shipping screening per model.*
- **Dashboard.** The `escape_info_label` says "Prediction Accuracy: …" but pulls from a rule-based confusion matrix (`get_escape_overkill_analysis`), not from `failure_probability`. The ML predictor's actual predictive accuracy is never shown.

### Learned sigma threshold

**Currently:** Applied to `TrackResult.sigma_threshold` after Apply to DB. Threshold line on Trends Standard detail sigma scatter. Recommended threshold + confidence + sample count in the ML Recommendations textbox (Trends detail mode).

**Where it should be but isn't:**

- **Analyze metrics tab and list items.** No "(ML)" vs "(formula)" tag. Operator can't tell if the threshold for this unit is data-driven or a fallback formula. The `threshold_method` field already exists in `ModelMLState`.
- **Settings ML training results.** Per-model line shows `T=0.00042 [PD] (67 samples)` only briefly post-training, only first 10 lines (settings.py L1264 `[:10]`).
- **Process page during processing.** No indication when a model uses formula fallback. Sigma_pass on those models is unreliable but the operator has no warning.

### Drift status

**Currently:** Trends Drift tab (CUSUM/EWMA chart per model), Dashboard Drift Alerts card (clickable, count + direction). QAAlert records with AlertType.DRIFT_DETECTED in Recent Alerts.

**Where it should be but isn't:**

- **Process page.** When a batch is processed, if it contains units of a drifting model, nothing tells the operator. "30 units of model 8340 just processed; this model has been showing quality degradation for 14 days."
- **Analyze list.** No drift indicator alongside the ML probability badge.
- **Drift response.** When drift detected, app does nothing except write a QAAlert and show a badge. No automated suggestion to retrain or investigate.

### Profiler insights

**Currently:** Top 3 ModelInsight strings in Trends Standard detail ML Recommendations textbox. Trim Difficulty tab uses profiler difficulty scores. "View All Details" dialog shows full ranking.

**Where it should be but isn't:**

- **Dashboard "Where to Focus" cards.** Use `get_linearity_prioritization` (pure SQL); ignore `difficulty_score` and `quality_percentile`. A model SQL-ranked #5 might be ML-ranked #1 by difficulty.
- **Cross-model comparisons.** `quality_percentile` computed across all models but never shown ("model X is bottom 10% for quality") — flattened into the single difficulty bar on Trim Difficulty.

## Recommended changes (prioritized)

### P1 — Batch at-risk summary on Process page  (S effort / L impact)

After `_on_processing_complete()` (process.py L773), the results list already holds AnalysisResult objects with track-level failure_probability. Add a counter in the processing loop: `self._high_risk_count += sum(1 for t in result.tracks if (t.failure_probability or 0) >= HIGH_RISK_THRESHOLD)`. Surface as one line in completion summary: "38 units predicted high-risk for FT failure — export list?" with a button that pre-filters Analyze.

**Directly addresses the highest-value ML use case (pre-shipping screening) with the smallest amount of new code.**

### P2 — ML accuracy feedback dashboard  (M / L)

Track failure_probability vs actual FT outcome. SQL join between TrackResult.failure_probability and linked FinalTestTrack.linearity_pass. New DB query method (S effort) + new card on Dashboard or tab in Settings (M to add UI).

Without this, the 134 trained predictors have no credibility signal. predictor_accuracy / precision / recall / f1 already exist in ModelMLState but only reflect hold-out accuracy at training time, not real-world FT correlation. Surface those training-time metrics in Settings as a table too.

### P3 — Freshness / method indicator in Analyze and Trends  (S / M)

Add small "(ML)" vs "(formula)" tag next to sigma threshold values. `threshold_method` is stored ('separation' / 'percentile' / 'weighted' / 'fallback'). Fallback = <20 samples = sigma_pass on that model is unreliable. Label-text change in `_update_detail_ml()` and `_create_analysis_list_item()`, no new data source.

Also: show `training_date` in Trends detail ML Recommendations ("trained 2025-11-03 on 67 samples" — currently only sample count is shown).

### P4 — Sortable failure_probability filter in Analyze  (M / M)

Add a sort-by dropdown to the Analyze filter row: "Date (newest)", "Status", "ML Risk (highest first)". When ML Risk selected, re-sort the already-fetched analyses list by max(t.failure_probability for t in a.tracks). Add "ML Risk >= __%" filter.

Enables pre-shipping screening by model.

### P5 — Drift → retrain prompt  (S / M)

When `get_drift_alerts()` returns non-empty + drifting models also appear in the staleness list, add a yellow banner: "2 models drifting and their ML may be stale — retrain recommended." Include "Go to Settings" button. Pure UI wiring of two data sources already loaded.

### P6 — Expose training metrics in Settings ML section  (S / S)

settings.py L1264 shows `T=threshold [P][D] (N samples)` per model. predictor_accuracy / f1 / auc already in ModelMLState. Add second line per model: `Acc=0.87 F1=0.82 AUC=0.91`. Answers "can I trust this model's predictions?" at a glance.

## Things to remove or simplify

- **`predict_with_confidence()` (predictor.py L414-467)** computes per-tree confidence intervals from the 100-tree ensemble. Never called anywhere. Either wire into the badge ("ML 73% [60-85%]") or delete to avoid maintenance burden.
- **HMAC-per-file integrity on predictor pickles (predictor.py L556-582)** is over-engineered for a local desktop app. The machine-local key means files are non-portable. Simplify to SHA-256 or remove integrity checks for local deployments.
- **PredictorConfig dataclass (predictor.py L56-67)** has hyperparameters never exposed in Settings UI. Either expose them or remove the abstraction.

## Things to automate

- **Auto-retrain trigger:** after Apply to DB completes and drift_alerts non-empty, automatically flag those models in `models_needing_data` with `retrain_reason=drift`. Next Settings visit shows "3 drifting models need retraining" prominently.
- **Scheduled staleness check:** `_refresh_staleness()` in Settings only runs when the Settings page opens. Move to app startup (after `load_all()`) and surface as a status bar indicator if any model exceeds 50 new records.

## Key files

- `ml/manager.py` — orchestrator; `apply_to_database()`, `get_drift_status()`, `_save_state_to_db()`
- `ml/predictor.py` — RandomForest per model; `predict_batch()`, `predict_with_confidence()` (unused)
- `ml/threshold_optimizer.py` — `_fallback_threshold()` (silent formula path), `method` field
- `ml/drift_detector.py` — CUSUM/EWMA, `drift_start_date`
- `ml/profiler.py` — `difficulty_score`, `quality_percentile`, `_generate_insights()`
- `ml/__init__.py` — `get_shared_ml_manager()` cache, TTL=300s
- `core/processor.py` L98-228 — ML threshold + predictor use during live processing
- `gui/pages/analyze.py` L737-800 — ML badge in list items (best current surfacing)
- `gui/pages/process.py` L619-712 — where batch-level ML at-risk summary is missing
- `gui/pages/settings.py` L183-275, L1198-1375 — ML section UI + training/staleness logic
- `gui/pages/trends.py` L1480-1526 (`_get_ml_recommendations()`)
- `gui/pages/dashboard.py` L645-669 (`_get_drift_alerts()`)
