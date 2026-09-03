# Taking V6 to work — first-day checklist

## 🆕 NEW LAPTOP, CLEAN DATABASE — this is the whole checklist (2026-09-02)

You said you'll reprocess every file fresh on the new machine. That is the
best possible path: **every data-repair script further down this page
becomes unnecessary**, because the fixed parser and analyzer produce
correct data on first parse — magnitudes recorded, clock times kept so
trim/FT links are right from the start, corrupt limit columns caught at
ingest, fail-point counts computed correctly. Skip those sections unless
you decide to keep an old database after all.

1. **Install Python 3.12 or newer** (3.13 is fine; this Mac runs 3.14). The
   pinned libraries — the exact set that passed 1,386 tests tonight — need
   3.11+, so the old "3.10+" note in the launcher was wrong and is fixed.
2. **Clone / pull `main`** — it is identical to `V6` now. Run `run_v6.bat`.
   First run builds `.venv` from `requirements-pinned.txt` (a few minutes,
   needs proxy/internet once).
3. **Settings → ingest folders**: add each laser folder and the Final Test
   folder, in the order you want them run. Once.
4. **HOME → "Process everything new"**, then walk away. The first run is
   the full history (~106k trim files, ~150k final-test files). The scan
   fast-path is in, but the first pass is still the long one — run it and
   let it finish; the summary line tells you folders · new files · time.
5. **Settings → train the per-model ML** after the ingest, so FOCUS drift
   baselines and the predictor exist on this machine.
6. Then the smoke pass: HOME shows FOCUS; Investigate → 8232-1 → stats
   table with banding; a unit → Final test overlay.

If anything in that flow feels slow, that is a bug, not a fact of life —
`scripts/ui_stall_probe.py` measures it; say what felt slow.

---
*Everything below is for a machine that keeps an EXISTING database.*

## ⚡ 2026-08-31 — 8232-1 never recorded HOW FAR out of spec it was (pull, then one script)

8232-1 recorded whether each track passed linearity and how many points
failed — but never **by how much**. The error magnitude was NULL on all 3,108
tracks since 2023, and on 5,506 going back to 2011. 8770 lost 361 the same
way. That is the zero-tolerance customer metric, missing on the model sitting
at #1 on FOCUS with a 57.8% escape rate.

Nothing looked broken, because every verdict column was intact. And it was
worse than missing: the app reads that column back as `abs(value or 0.0)`, so
a dropped magnitude re-entered charts and exports as **0.000 — a flawless
part**.

**Cause.** These files open with a six-point unmeasured lead-in, so the error
array starts with NaN. The magnitude was taken with `max(abs(e) for e in
errors)`, and Python's `max()` returns NaN when the *first* element is NaN —
a NaN anywhere else is skipped and the answer comes out right. Pure positional
lottery, and the DB agrees exactly: across 42,387 tracks since 2023, NaN at
index 0 → magnitude NULL (3,314 tracks), NaN anywhere else → stored fine (2).

**Pulling fixes every file processed from now on.** The measurements were
never lost, so the old blanks are recomputable exactly from arrays already in
the DB:

> **✅ ALREADY RUN on this machine's `data/analysis.db` (2026-08-31).** Backup
> at `data/analysis.db.bak-2026-08-31-pre-linerror-fix`. Verified after:
> magnitudes 92,819 → **100,585** (+7,766 exactly, matching the dry run),
> 8232-1 4,035 → **9,541** (+5,506), **FAIL count unchanged at 29,455 — no
> verdict moved**, `PRAGMA quick_check` ok. The sweep check it was failing now
> passes. You do NOT need to run it again here. Run it on the WORK machine's
> database when you next pull there — these commands, in this order:

```bash
cp data/analysis.db data/analysis.db.bak-2026-08-31-pre-linerror-fix
python scripts/backfill_linearity_error.py --dry-run
python scripts/backfill_linearity_error.py
```

Recovers 7,766 magnitudes across 20 models (8232-1: 5,506). Rehearsed on a
copy of the work DB — 400 recovered values checked against a fresh analyzer
run, all identical to the last digit. It writes magnitude columns only, so
**no verdict moves**; the code change was separately confirmed not to flip a
single pass/fail across the 2,439 tracks it can touch.

Expect 8232-1 to look bad rather than blank: median 2.3x spec, landing in the
same distribution as the tracks that never broke. That is the real number, and
it has been invisible for years.

The script also names 1,842 older tracks (8084-2, 7458, 8531-1 — all pre-2023)
it deliberately leaves alone: their `optimal_offset` is missing too, so they
need reprocessing, not a backfill.

### One number will move if you ever reprocess those older tracks

Those same 1,842 tracks carry an **inflated fail-point count** today. When the
offset came out NaN, every graded point compared as a failure, so the DB holds
things like **71 fail points where the truth is 13**. The verdict was right
(they really do fail) but the count is manufactured. Reprocessing corrects it;
the backfill deliberately does not touch it, so nothing changes under your feet.

Checked the whole blast radius against the work DB before committing: all
9,633 NaN-carrying tracks re-run — **0 verdict changes, 0 tracks newly marked
ERROR** — and 1,867 fail-point counts corrected downward (1,842 of them the
NULL-offset set above, plus 25 that shift by one). On 2,317 NaN-free control
tracks the new code is byte-for-byte identical to the old.

The sweep now fails if any model ever grades tracks it cannot measure again.

## ⚡ 2026-08-30 night — trim-vs-FT numbers change, and a new stats table

### 1. Trim-vs-FT was reading the wrong file — the pull fixes most of it

A unit that fails linearity gets re-trimmed until it passes, all on the same
day, and a two-track unit writes one file *per track*. The app was reading a
single file for the unit's verdict, which went wrong two ways: it could link
an early failing attempt (scoring a good unit as wrongly rejected), and on a
two-track unit a passing Track B could hide a failing Track A. Linearity is
zero-tolerance, so that second one was calling failed units passed.

**Just pulling gets you most of the correction** — the "every track must
pass" rule needs no repair at all. The numbers, company-wide over 12 months:

| | overkills | escapes |
|---|---|---|
| before | 1257 | 761 |
| **after pulling** | **1304** | **783** |
| after also repairing | 1238 | 806 |

Not all in the direction you'd guess: **6607's overkills go UP, 415 → 486**,
because it's mostly dual-track and its failures were being masked. **8508
gains 20 real escapes** (30 → 50) that were being hidden as agreement.

### 2. Optional, when you want the last few percent

```bash
python scripts/repair_trim_ft_links.py
```

About six minutes, and it rewrites ~106k rows, so it's your call whether to
run it on the production database. All it adds is correct *ordering* among
same-day attempts — worth a further 66 overkills and 23 escapes company-wide.
Per model the difference is small: 6607 486 → 484, 8340-1 135 → 132, 8232-1
escapes 614 → 619. Look at the corrected numbers first and decide.

Until you run it, two QA sweep checks fail by design, naming this script as
the remedy. That's expected, not a broken build.

### 2. INVESTIGATE — the stats table that replaces the Excel round-trip

On the Model page: **n · avg · min · max** for untrimmed and trimmed
resistance, electrical angle, linearity error, margin to spec and sigma
gradient — split **ALL units | LIN-PASSING** (lin-passing = PASS + WARNING,
your rule: linearity is the disposition, WARNING is the internal sigma
watch). There's a **lot selector** next to the model and window pickers, and
picking a lot adds that lot's values beside the historical band with a
plain-English verdict per metric. "Export model to Excel" gains a **Stats
table** sheet that mirrors the screen exactly, so what you show an engineer
equals what you saw.

**Read the small print on two rows.** "Tracks that passed linearity" and
"Tracks already in spec before trim" count **tracks, not units** — they're
named that way on purpose. On 6607, 1,087 of 4,841 unit-days have one track
passing while another fails, so a track rate and a unit yield are genuinely
different numbers. Unit yield lives on the Dashboard, computed properly.

**Every cell shows what it threw away.** Your resistance columns contain
readings of 1e12 Ω and 0 — only 38 in the whole database, but on 6607 seven
of them dragged the average from a true 4,282 Ω to 32,079 Ω. Those are now
excluded per model, with the count shown. Same for 94 rows storing a sigma
gradient of 999.999, which turned out to be a marker on records whose
processing errored — they made 8856's average read 433.5 instead of 0.001.

**One thing I inferred, so check it:** "already in spec before trim" means
`untrimmed_error_max <= linearity_spec`. That's my reading of the columns,
not your words. It reads 17 of 9,922 on 6607.

## ⚡ 2026-08-30 evening — FOCUS rows now say WHY, and flag mismatched specs
Same pull, no schema change, nothing to retrain. Three things look different
on screen.

1. **Rows say what's driving them.** Each FOCUS row carries a short tag from
   the drift watch naming the metric most likely behind the excursion —
   so the row tells you where to look, not just that something moved.
2. **A blip that recovered ranks below a fire that's still burning.** If a
   model has run at its own baseline since its last out-of-control lot, the
   row says **"has run at baseline since"** and its rank is divided by the
   number of clean lots since. The failing-units-per-week number you see is
   still the real measured one — only the position in the list is discounted.
   Live example: **6126** blipped once on Jul 31 (33.3% against a 33.2%
   limit) and has run at baseline since; it sat at #3 and now sits at #4,
   below everything actively burning.
   Read "at baseline" literally — it means *back to its own normal rate*, not
   *good*. **8340-1** says it while running 73%, which is exactly its normal.
3. **NEW — the app tells you when trim and final test aren't grading the same
   thing.** Where a model's two stations hold different limits, the Model page
   shows an amber line under the verdict, and the FOCUS row is marked
   **⚠ trim/FT specs differ**. This matters because every cross-station number
   the app prints (escapes, overkills, the Gap) assumes both stations ask the
   same question — where they don't, an "escape" isn't a missed defect, it's
   two different requirements being subtracted.
   **Four of the eleven models on FOCUS are flagged right now, including the
   top two:** 8232-1 (100% of shared positions graded differently), 6126
   (96%), 8340-1 (24%), 6607 (21%).
   This is **visibility only** — nothing was re-graded, no verdict moved, and
   the app does not pick a "correct" spec. What to do about the mismatches is
   your call, and it's the biggest open question on the app right now.
4. **Your turn:** open 6126 and 8340-1 on the Model page. Those two rows are
   where all of this shows at once — the recovery marker, the driver tag, and
   the spec banner. Tell me if any sentence reads wrong.

## ⚡ 2026-08-30 — Triage's card wall is now FOCUS (do nothing but pull)
1. `git pull`, then launch as normal. No new database schema, nothing to
   retrain, nothing to click — the drift detectors you already trained keep
   running exactly as before.
2. **Triage's card wall is gone.** The top of the page is now
   **FOCUS — drifting now, biggest first**: a ranked list of models, biggest
   problem first, measured in extra failing units per week instead of an
   alert tier. A model only shows up while an out-of-control lot sits inside
   its last 5 lots — and it drops off the list BY ITSELF once a few lots run
   clean. No dismiss button, no queue to clear, nothing to acknowledge.
   Underneath FOCUS, a smaller "chronically high, stable" strip holds models
   that run a bad rate consistently but aren't currently drifting — that's a
   different problem (capability, not drift), so it's kept out of FOCUS
   instead of re-alarming on it every morning.
3. **The Model page headline chart changed.** It now opens on the SPC lot
   chart — fail rate per production lot, with a shaded band showing what
   that model's own history says a lot that size can do by chance, and
   flagged lots annotated in plain English. The old unit-dot chart is still
   there, one click away behind a **Lots · SPC / Units** toggle at the top
   of the chart.
4. **Evidence Excel export gains a sheet.** "Export evidence pack" now
   includes a "Lots (SPC)" sheet that mirrors exactly what the FOCUS row and
   the Model-page chart show on screen — same numbers, so the export can
   never contradict what you saw in the app.
5. **Your turn:** do one manual smoke pass after pulling — open Triage,
   click a FOCUS row (it should land you on that model's page, focused on
   the metric that triggered it), then toggle Lots/Units on the Model-page
   chart. That's the only step left undone; it needs eyes on the real GUI.

## ⚡ 2026-08-29 — scanning is fixed (do nothing but pull)
The Final Test folder took 68–75 minutes to scan every time because the app
re-read all 171,000 files to identify them. It no longer needs to.

1. `git pull`, then launch as normal. The database upgrades itself at
   startup — no buttons, no terminal.
2. **The first Final Test scan after the pull is a one-time verification
   pass.** The progress bar says "Verifying changed/legacy files… X/Y".
   Expect roughly 15–30 minutes on the share (it's the old work, done once,
   8 files at a time instead of one). Let it finish.
3. **Every scan after that costs about walk-time — a minute or two** for the
   full FT folder. Small daily folders are near-instant. Nothing to press:
   the records stamp themselves during that first pass.
4. The log now prints one line per batch with every phase and its cost:
   `Batch phases: walk 41.2s (171,006 files) | load 2.1s | check 1.4s
   (170,940 known in memory) | verify 66 files 8.2s | process 66 files 240s |
   rematch skipped (no new trims) | advance 3.2s`
   If anything is ever slow again, that line names the phase — send it over
   and it's diagnosed in one read. Per-file lines moved to DEBUG, so the
   5 MB log now holds days instead of three hours.

## ⚡ 2026-07-14 morning (after tonight's push, commit 3d894fa)
1. `git pull` on main.
2. Settings → ML Training → **Retrain drift detector** once (~1 min). The
   watch now covers 12 metrics including two FINAL-TEST ones: FT lot fail
   rate and escape rate (trim PASS → FT FAIL). Models that only ever appear
   in final-test data train too.
3. Process any folder as normal. The first batch auto-relinks final-test
   records to their trims (the log prints "Unlinked-FT rematch: N of M
   linked" — expect a large N once, since the 7-13 rebuild processed
   everything in one batch and FT files that arrived before their trims
   stayed unlinked). The match window is now 180 days (real trim→assembly→
   test dwell), so link rates jump.
4. What you'll see: Model + Triage pages now separate "WHAT THE APP IS
   TELLING YOU" (verdict, pills) from "WHAT YOU'RE LOOKING AT" (chart,
   units); the drift table groups metrics into process signals / trim
   outcome / final-test outcome; fail rates read as percentages.
5. NEW (7-14 evening): the Model page verdict now answers the wasted-laser
   question — "⚠ N% of trimmed units already met linearity BEFORE trim."
   Work-DB standouts to check first: 8877-4 (84%), 8167 (84%), 8436 (65%),
   8755 (59%), 8863 (50%). Those are candidates for raising the as-fired
   resistance target so the trim disappears. Also: unit charts now show the
   pre-trim dotted line, clicking a Final Test unit opens its sweep, and
   the Dashboard scrolls.

The app is finished and verified on the home copy of the database. Because
your WORK database is a separate copy, three one-time steps run there on day
one. Everything is a button — no terminal needed.

## 1. Get the code onto the work machine
Copy the whole project folder EXCEPT `data/` (or `git pull` if the work
machine tracks the repo). Two rules:
- **Never bring the home `data/analysis.db` to work** — the work database is
  the live one; keep it. Put it at `data\analysis.db` inside the new folder
  (create the `data` folder and copy the DB file over from the old install).
- Don't copy `.venv` — it gets rebuilt on the work machine.

Launch with **`run_v6.bat`** (new, replaces `run_analyzer.bat`): it has no
hardcoded paths, builds `.venv` and installs dependencies on first run, and
starts the V6 UI. The old `run_analyzer.bat` pointed at the 3.0.2 folder and
launched V5 — don't use it.

## 2. First launch (automatic)
The app upgrades the work database by itself the first time it opens:
- adds the new drift-tracking columns,
- tags any LTS3 (third laser) files already in the database as System C.
Nothing to click; it logs what it did.

## 3. Two one-time buttons (Settings page)
1. **Database Cleanup → Recompute unit statuses** — re-grades history under
   the correct rule: linearity decides pass/fail (zero tolerance); sigma is
   only a "watch" flag, never a rejection. On the home copy this re-labeled
   7,840 units that were linearity failures hiding as Warnings. It shows you
   the counts and asks before changing anything.
2. **ML Training → Retrain drift detector** — rebuilds the drift baselines
   with all of this week's fixes (corrupt readings can no longer own the
   alerts; sudden-jump detection now works after restarts).

Done. Process files as normal — Triage now updates itself after every batch.

## 3½. After pulling the 2026-07-10 fixes: rebuild the venv ONCE
Delete the `.venv` folder in the project directory, then launch `run_v6.bat`.
It rebuilds with PINNED library versions — the exact set proven at home.
(Friday's processing failure was an unpinned install resolving a newer
pydantic with different behavior. Pinning ends that class of surprise.)
The app also self-checks its environment at every launch and writes
"Environment OK — …" (or a plain-language failure) at the top of the log.

## 4. Two things to watch the first week
1. **First LTS3 batch.** LTS3 support is tested against synthetic files but no
   real third-laser file has been through it yet (no sample existed). Process
   the first LTS3 folder, then check the unit shows System C on the Model
   page. If a file errors, it will say so loudly — send it to Claude.
2. **Fix the mislabeled FT file.** `8074-sn4 blue-1_12-18-2026_9-55 AM.xls`
   (model 8074, SN 4) carries a future date from a typo in the filename. The
   dashboard excludes and flags it; correct the filename to the real test
   date and reprocess (or delete the record). The ⚠ note clears itself.

## What changed since the version you had at work
- **Processing is fast again.** The "checking against database" hang is fixed
  (the scan no longer re-reads every known file over the network).
- **No more freezes.** The intermittent lock-ups had a real cause (background
  work touching the screen); that whole class of bug is gone.
- **Yields tell the customer truth.** Headlines are linearity yield
  (pass + watch), with the clean-pass/watch/fail split underneath. The
  company trend chart on the Dashboard shows it by week/month, split by
  laser system — including the new LTS3 laser as its own line.
- **Third laser (LTS3) supported.** Files under an LTS3 folder are tracked
  as System C automatically. Keep LTS3 files under their LTS3 folder — the
  folder name is how the app knows.
- **Charts read correctly.** Units draw as dots with a daily-median trend
  line; suspect readings are clamped to the edge and counted, never hidden;
  every chart says how fresh its data is.
- **Alerts are trustworthy.** Clicking an alert always lands on the model's
  data (never an empty chart); one corrupt reading can't fake a +16σ alarm;
  σ is explained in plain language on the Model page.
- **Investigation tools.** Unit chart has a track selector for multi-track
  units; "Save chart…" produces the print-ready document with unit info,
  metrics, and a status box (like V5's export, with the status rule fixed);
  "Export evidence pack" gives the model's full history in Excel with a
  monthly better-or-worse summary.
- **Dashboard Gap column.** Trim% − FT% per model: big negative = rejecting
  good product at trim (overkill), positive = escapes reaching final test.
- **Final test overlay, on the unit chart.** Open any unit and flip the
  "Final test overlay" switch: the linked final-test sweep is drawn over the
  trim trace in orange, with its OWN correction and its OWN spec limits (the
  FT station makes its own adjustment — grading it with the trim's would draw
  a curve nobody measured). The caption names the linked file's date and the
  match confidence. If the unit was final-tested more than once, you get the
  NEWEST test and the chart says "newest of N". "Save chart…" includes the
  overlay when it is on. When there is nothing to show — no link, a match
  below 0.70, no stored sweep, or an FT file whose position column isn't a
  position — the switch is greyed out and the caption says which of those it
  is, instead of a blank chart. This was the last thing V5 could do that V6
  could not.
- **Unmeasured points are visible.** A point the tester recorded no value for
  is counted as a failure (a zero-tolerance spec can't call an unmeasured
  point good) — it now shows as a hollow purple square on the axis line at
  the position it belongs to, so the marker count on the chart matches the
  fail count beside it. The document adds "Unmeasured: N (counted as fail)".

## If something looks wrong at work
Tell Claude what you saw — screenshots help. Two self-check programs ship in
`scripts/` and run automatically during development; you never need them,
but they're why each change arrives verified.

## Known items still on the list (not blockers)
- ~~Trim-vs-FT overlay chart inside the unit view~~ — **done 2026-08-31**, see
  above. V5's Compare page no longer holds anything V6 lacks; V5 remains
  launchable as the fallback until you have used the V6 overlay at work.
- ~~Fix Missing Tracks~~ — **closed 2026-08-31.** The tool now lives in V6
  (Settings → Database maintenance), so no V5 fallback needed. What it will
  actually find on your database: **264 final-test + 3 trim records** with no
  stored sweep, re-parseable only at work (the source share). The long-carried
  "NULL-array rows" worry turned out to be by-design: those tracks are the
  untrimmed second tracks of 2-track units and still hold their pre-trim
  sweep. Zero gradeable tracks lack their waveform — now pinned by the
  self-check so it stays true.
- The 2.1e8 Ω-style historical bad readings get flagged at ingest going
  forward; the stored ones are excluded-and-disclosed by the stats table.
