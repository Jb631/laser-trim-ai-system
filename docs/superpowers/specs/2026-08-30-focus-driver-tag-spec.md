# FOCUS row "likely driver" tag — approved spec (2026-08-30)

James's live smoke-test feedback on the shipped FOCUS list: "it doesnt tell me
if its resistance, linearity or why the model is flagged." Approved fix: each
FOCUS row carries a third line naming the worst-moving PROCESS metric.

## Requirement

- `ml/spc.py`: `FocusEntry` gains `driver: Optional[str] = None` — preformatted
  plain language, e.g. `untrimmed resistance ↑ (+2.1σ vs its baseline)`.
  Direction arrow from the shift sign; names via `metric_label`
  (ml/drift_types.py). None when no process metric is flagged.
- Source: the EXISTING drift-detector state (read-only — never retrain).
  Read ml/manager.py (get_drifting_models / get_model_drift_status /
  ModelAlertSummary) and pick the cleanest read API. Per-model calls are fine:
  enrichment runs AFTER membership/ranking, only for the ≤ ~17 listed models.
  Enrichment failure must never break the list: per-model try/except →
  driver=None + one logger.exception.
- EXCLUDE outcome metrics from candidacy (anything in FRACTION_METRICS — the
  row already IS the outcome). Driver = worst currently-flagged process metric
  by |sigma shift| (or the API's severity order if shift unavailable).
- Chronic entries: driver stays None; no driver line rendered (their verdict
  already says capability-not-drift).
- `gui/v6/widgets/focus_list_zone.py`: FOCUS rows get a third small line
  (TEXT_SECONDARY, one size below line2): `likely driver: {driver}` when
  present, else `driver unclear — open the model`. Extend `focus_row_texts`
  with `line3` (testable); render in `_FocusRow`; chronic rows unchanged;
  click-binding must cover the new label; row grows ~16-20px (scroll absorbs).
- Model page routing unchanged.

## Process rules

- `.venv/bin/python`. TDD: tests first —
  tests/test_spc_db.py: seed detector-state rows so (1) a driver IS chosen,
  (2) an outcome metric is NEVER chosen, (3) an unflagged model yields None;
  tests/test_focus_list_zone.py: line3 wording both cases + absent on chronic.
  Red for the right reasons, then green; full suite junit-counted.
- scripts/app_qa_sweep.py FOCUS section: new check — every focus entry's
  driver is None or names a NON-fraction watched metric; must FAIL on
  exception. Real DB is at data/analysis.db (James's 3.6GB copy) — run the
  FULL sweep directly and record tally + FOCUS lines. The sweep regenerates
  docs/v6_design_review_2026-07-07/qa_sweep/qa_evidence_6607.xlsx — commit the
  churn (established precedent). Do NOT delete data/analysis.db (it is real
  data now, not the stray).
- One commit: "FOCUS rows say WHY: likely-driver tag from the drift watch
  (James's live ask)". Push after verification.
