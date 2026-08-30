# Taking V6 to work — first-day checklist

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

## If something looks wrong at work
Tell Claude what you saw — screenshots help. Two self-check programs ship in
`scripts/` and run automatically during development; you never need them,
but they're why each change arrives verified.

## Known items still on the list (not blockers)
- Trim-vs-FT overlay chart inside the unit view (V5's Compare page still has
  it; V5 remains launchable).
- Fix Missing Tracks for records with no stored sweep (5,379 units are
  un-gradeable until re-parsed; they're skipped, never guessed).
- The 2.1e8 Ω-style historical bad readings get flagged at ingest going
  forward; a cleanup pass for old ones comes with Fix Missing Tracks.
