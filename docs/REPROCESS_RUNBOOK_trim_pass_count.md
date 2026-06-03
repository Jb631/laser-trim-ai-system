# Reprocess runbook — populate `trim_pass_count` (work machine only)

`trim_pass_count` is the only composite trim-risk feature that is **not derivable**
from stored data — it is counted from the trim sheets in each source file. The
current parser populates it correctly (verified: System A → `2`, System B → `1`).
The production DB has it NULL only because those rows predate commit `26894cb`
("capture and surface trim_pass_count per track").

The other three trim-effort features (`untrimmed_error_max`, `untrimmed_rms_error`,
`resistance_change` / `resistance_change_percent`) are derivable from columns
already in the DB and are filled instantly by `scripts/backfill_trim_effort.py`
— **no reprocess needed** for those.

> **Environment note:** use `python3` (there is no `python` on PATH on the dev
> machine; adjust to your work machine's interpreter).

## Steps (work machine, against the REAL DB)

1. **Pull `main`** — brings Phase 1 (Tasks 1–3): the `untrimmed_error_max` column +
   its migration, and the derivable-feature backfill script.

2. **Back up the DB first:**
   ```
   copy data\analysis.db data\analysis.db.YYYY-MM-DD.bak      (Windows)
   cp   data/analysis.db data/analysis.db.YYYY-MM-DD.bak      (macOS/Linux)
   ```

3. **Run the derivable backfill** (instant, no reprocess). This also lets the app's
   `_run_migrations()` add the `untrimmed_error_max` column to the existing DB on
   first connect:
   ```
   python scripts/backfill_trim_effort.py data/analysis.db
   ```
   Expected fill rates (bounded by usable untrimmed data, ~the same as
   `untrimmed_sigma_gradient`): `untrimmed_error_max` / `untrimmed_rms_error`
   ≈ 82%, `resistance_change` / `resistance_change_percent` ≈ 93%. Re-running is
   safe (idempotent — second run reports `0 array-derived row-updates`).

4. **Reprocess source files so `trim_pass_count` is captured.** Either:
   - **Full reprocess** of the archive, OR
   - **Incremental:** the content-hash dedup (Batch D) skips unchanged files, so a
     reprocess only re-reads files whose rows still need it. Before running, confirm
     the processor maps `trim_pass_count` — it is mapped in
     `src/laser_trim_analyzer/database/manager.py` `_map_track_to_db`
     (`trim_pass_count=getattr(track, 'trim_pass_count', None)`).

   > Note: a full reprocess regenerates `trim_pass_count` for every row; the
   > incremental path only fills rows whose source file is re-read. If you need
   > `trim_pass_count` for **all** historical rows, do the full reprocess.

5. **Verify:**
   ```
   SELECT COUNT(*) FROM track_results WHERE trim_pass_count IS NOT NULL;
   ```
   Expect the large majority populated. Spot-check the distribution (values like
   1, 2, 3…); a value of `0` means a test-sweep-only file (no trim run), which is
   valid.

## Gate

The Phase 2 deploy-gate re-validation (`scripts/validate_composite_deploy_gate.py`,
Task 11) must be run **after** this reprocess, because the third orthogonal failure
mode (trim headroom = `trim_pass_count`) only exists once it is populated. Until
then, Phase 2 trains on the available features and the gate deploys the composite
only where it already beats `untrimmed_error_max` alone.
