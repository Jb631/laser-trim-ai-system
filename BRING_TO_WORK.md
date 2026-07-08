# Taking V6 to work — first-day checklist

The app is finished and verified on the home copy of the database. Because
your WORK database is a separate copy, three one-time steps run there on day
one. Everything is a button — no terminal needed.

## 1. Get the code onto the work machine
Copy the whole project folder (or `git pull` if the work machine tracks the
repo). Launch as usual — Windows: `deploy.bat` / `run_analyzer.bat`; the V6
UI starts with the `--v6` flag (same as you've been using).

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
