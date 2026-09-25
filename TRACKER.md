# Tracker

What is open, in what order, and who holds the next move. Updated in the same
commit as the work it describes. `BRING_TO_WORK.md` stays the place for
step-by-step instructions at the work machine; this is the index above it.

Last updated: 2026-09-24

## ▶ Where things stand (Claude, 2026-09-25)

**On `main`:** everything through this push (2026-09-25 morning) — facelift step 2 (the other six
pages, and the Units chart redrawn) after a whole-branch review, its fix round and a re-review; the
database guard (only the app opens its default database — every script now names one); and three
small fixes. Before it, `422da7a` (2026-09-24): facelift step 1 and every parse fix. At work:
`git pull`, then the top section of `BRING_TO_WORK.md` (and the two 2026-09-24 sections if you have
not pulled them yet). Nothing needs running except the optional TrimVolts back-fill. **Being built
now:** the rest of the ingest-speed work (A4/A3/A5, Tasks 5–12: the batched writer, worker
processes, workers that come back) — the save probe you run at work (A6) tunes it. Pushed on
2026-09-25: the facelift follow-ups (F4) in the morning; the rest of the findings catalogue (B6) at
midday (after pulling it, refresh the findings once: Settings → Database → Refresh process findings);
the save probe; and in the afternoon F5 ("Inactive" models) with the first three speed steps.

**The rebuild is done** (B2: 168,501 files in 21.8 hours, finished 2026-09-22) and every number
that rested on a final-test verdict has been re-derived on it (B3).

| Workstream | State | Next move |
|---|---|---|
| **A. Processing speed** | designed 2026-09-25: the save is NOT the bottleneck; worker processes overlap parse and save | run the save probe at work (A6, yours); then Tasks 2–12 — Claude |
| **B. More useful information** | rebuild done; the findings catalogue complete — 11 analyzers (B6, 2026-09-25); laser 1's TrimVolts captured | the back-fill (James, optional) → B7 cut-length model |
| **C. Review and refactor** | the review is done (C1); C2 step 1, the database guard and three parked fixes shipped | C2 step 2 (`database/migrations.py`) — Claude |
| **D. Checks at the shop** | D1, D3, D4, D5, D6, D7 open | James |
| **E. Backlog upload** | shipped (E1) | — |
| **F. Facelift** | steps 1 and 2 shipped; F4 follow-ups shipped | F5 "Inactive" models (+ F4's review gaps) — Claude |
| **G. Parse fixes** | all built, reviewed and pushed (`422da7a`) | — (the back-fill is yours, optional) |

### One thing of mine to own (2026-09-24): the home database was written to

Twice today a verification script run by one of my subagents opened `data/analysis.db` on the Mac
READ-WRITE — the file I had undertaken never to write. Each called the processor on a test file
without redirecting its database first, and the processor looks up model specs in the app's
default database, which on the Mac is that file. Opening it ran the app's start-up migrations.
**What changed** (every figure from read-only queries): five new, EMPTY columns (`error_reason`,
`increment_volts`, `increment_volts_first_row`, `increment_volts_truncated`, `initial_trim_value`)
and two bookkeeping rows in `app_meta`. **Nothing else:** no row in any table carries that day's
date; the ERROR (237), Unknown-model (381) and laser-3 (547) counts equal what was measured before;
the start-up unit_id backfill provably had nothing it could change; SQLite's own `quick_check`
reads "ok". These are the same changes the app makes by itself on its first launch with this code.
**Your WORK database is untouched** — this was the home copy. **What I did about it:** the Mac file
is now read-only (`chmod a-w`), so a stray write fails loudly instead of landing, and every brief
now explains exactly how the processor reaches the database. **Before you run the app on the Mac,
give it write access back: `chmod u+w data/analysis.db`.** It opens read-only now, but this copy
lacks the one column the newest code adds (`track2_parameters`), so Findings cannot refresh until
a writable launch adds it (seconds). (I first wrote here that the app would open it read-only; the
final review proved it could not — my check had used an empty database, and an unguarded start-up
backfill stopped the app. That backfill is guarded now, `9d10803`.) I put the write access back
myself when this session's subagent work is over. **Since 2026-09-25 the code itself prevents a
repeat** (`ffd2516`): outside the app, asking for the database without naming one refuses before
any file is opened, and says so once — so a script like those two stops with a message instead.

### Decisions that are yours

**New since 2026-09-23** — D4 (what the old template's `Start Point` / `End point` means), D5 (what
"Micro-Lin max. error slope" is for), D6 (were the 8506A/8506B limits loosened by ECN?), D7 (the work
laptop's screen resolution and Windows scaling), D8 (which element 8397-2's "Section 3_0 Test" files
test) — all in section D. Plus D1 (which 8232-1 limit table is the intended one), D3 (the QA sweep's one remaining
red line) and H4 (retire V5).

**Still open from 2026-09-20** (none answered yet):

1. **Ungraded final tests count as PASS** at file level and FAIL at track level. Small overall (0.19 %),
   large per model: 8502 reads 27.6 % FT pass, 15.5 % over files that were really graded.
2. **One definition of yield.** The Excel export says 12.3 %, the app says 76.0 %, same data (the
   export counts WARNING as a failure; your rule says sigma is never a rejection).
3. **"Fix Missing Tracks" writes invented numbers** (sigma 0, spec 0.02). Proposal: write blanks.
4. ~~**Five backfill scripts write to the work database by default**~~ — **settled by the guard
   (2026-09-25, `ffd2516`), yours to undo:** each script now needs the database named
   (`… data\analysis.db`); `backfill_trim_effort.py` still has no dry run (its header says back up first).
5. **A full final-test rematch can fire at app startup** after a migration — the shape of the 09-14 night.
6. Sidebar: **Findings** sits third, after Investigate. Say if you would rather it lived on Home.
7. Backlog prices are MERGED on upload (a model that drops off the backlog keeps its last price).
8. A no-cut check sweep is a real measurement of the unit at that moment. Should it count for anything?

## Which laser is which

The shop's numbers do NOT follow the code's letters (James, 2026-09-20):

| Shop name | Folder | Code | Records |
|---|---|---|---|
| **Laser 1** | `LTS` | System B | each pass's sweep + that pass's settings |
| **Laser 2** | `DLTS` | System A | the same, **plus the cut applied, trim current, target output and measured output at every position** |
| **Laser 3** | `LTS3` | System C | laser 2's sheet format (not laser 1's — corrected 2026-09-24; pinned by `tests/test_laser3_is_read_like_laser2.py`), including the per-position cut data |

## A. Processing speed

A full rebuild reported 158 hours. The laptop and the parser were never the
problem — per-file conversations with the share were. Same code, same laptop:
14 files/sec on local files, 0.5 on the share.

- [x] **Read each file once, not six times** — `0add30e`
- [x] **One file-info lookup per file, not twelve** — each costs 113 ms on the
      share — `74924ae`. Together: 1907 → 976 ms/file, measured on the laptop.
- [x] **…then corrected the same day** — `30c66ff`. The first version shared a
      lookup for ten seconds, which let a just-rewritten file look unchanged and
      be skipped; two existing tests caught it. Sharing is now per parse, never
      by clock. Lookups per file: 12 → 4 (the unsafe version reached 2). On the
      VPN expect ~1.2 s/file rather than the 0.98 s measured before the repair.
      Reaching one lookup safely means carrying a read-time snapshot through
      the processor — do it inside A3, where that code is opened anyway.
- [x] **Speed probe** — `scripts/ingest_speed_probe.py`, times each layer
      separately, writes only to a throwaway database.
- [x] **Ping through the tunnel: 54 ms, no loss** (2026-09-20). A file lookup
      costs 113 ms = **two round trips + ~5 ms of server time**. So the server
      and SentinelOne are NOT the slow part; distance is. The VPN is not narrow,
      it is 54 ms away, and the app made ~45 round trips per file (now ~17).
      Claude's earlier "20 Mbps bandwidth ceiling" is withdrawn as unproven —
      robocopy's 2.5 MB/s may be a one-connection limit, not the pipe.
- [ ] **A1 · Run the probe AT WORK, on the office network.** *James · 1 minute.*
      **Prediction, now strong:** round trips at work are under 1 ms, so lookups
      should read ~1–2 ms and the app should run near 14 files/sec.
      `cd C:\dev\laser-trim-ai-system` first, then
      `.\.venv\Scripts\python scripts\ingest_speed_probe.py "\\192.168.66.9\BTXData\Departments\System Data\TEST_DATA\DLTS" 25`
      Read the `stat` figure on the first line:
      **~1 ms** → the VPN was the limit; rebuild at work, expect 5–6 hours.
      **~113 ms** → the file server (or SentinelOne's handling of the share) is
      slow even on the LAN; raise it with IT and go to A2.
- [ ] **A1a · Is the share making PARSING slow?** *James · 2 minutes, safe to run
      while an ingest is going.* `scripts/parse_locality_probe.py` times the SAME
      files where they live and again as local copies. **Why it exists:** the work
      laptop parses at 231–293 ms/file where the Mac does 67.8 ms on comparable
      files, and a 4× CPU gap on a 48 GB laptop is not credible. `raw read` prices
      ONE read; parsing opens the workbook AGAIN through pandas and walks a dozen
      sheets, so anything charging per OPEN (SMB round trips, on-access scanning)
      hides inside the parse number.
          `.\.venv\Scripts\python scripts\parse_locality_probe.py "\\192.168.66.9\BTXData\Departments\System Data\TEST_DATA\DLTS" 25`
      Local much faster → A2 (mirror, ~73 GB). No difference → A3 (processes).
- [x] **A1a · Is the share making PARSING slow? NO — measured at work 2026-09-21.**
      `parse_locality_probe` on 25 real files: on the share 172.8 ms/file, local copies
      **201.1 ms/file — local was SLOWER (0.9x)**. So the share and the scanner are
      cleared, and **A2 is dead: mirroring 73 GB would have bought nothing.** (James:
      "im not copying all the files to the computer thats crazy" — and he was right.)
      Parsing there is CPU-bound at ~173–293 ms/file against the Mac's 67.8, which is a
      real single-core gap between the machines, not a defect.
- [ ] ~~**A2 · Local mirror of the share**~~ **— RULED OUT** (A1a above: local is no
      faster than the share). Kept only so nobody proposes it again. — now UNLIKELY to be needed (the ping
      cleared the server); only if A1 still shows slow lookups on the LAN.
      The idea was one `robocopy /MT:32 /Z` copy, then rebuild at local speed. The
      measurement says there is no local speed to gain.
- [ ] **A6 · Run the save probe at work — James, ~6 minutes** (the top section of BRING_TO_WORK):
      it measures the laptop's own save and flush cost and where the loop's time goes, on a
      read-only copy, and answers the design's one open question (below). Paste the whole output
      back, with the three questions there.
- [ ] **A3 / A4 / A5 — DESIGNED AND PLANNED 2026-09-25**: spec
      `docs/superpowers/specs/2026-09-25-ingest-speed-design.md`, plan
      `docs/superpowers/plans/2026-09-25-ingest-speed.md` (12 tasks). **What it measured, which
      corrects A0 below:** a save into the 6 GB database costs 4–5 ms — the batch line's "save" was
      mostly waiting for Python's lock while four parser threads ran (33 ms wall, 5 ms CPU); A0's
      339 ms "parse" came from probes whose throwaway database had no model specs (the real analysis
      costs ~48% more); and with worker PROCESSES the parse and the save overlap instead of adding,
      so A0's 1.35x cap does not hold — a prototype (8 workers, one consumer saving 20 files per
      transaction) ran laser-2 files at 13.3 ms/file against 102–116 today, on the Mac. Every
      final-test save also scans the whole final-test table (no index on `file_hash`: 22 ms → 3.5 ms
      with one). About 200 ms/file of A0 does not reproduce off the laptop — the probe (A6) says
      whether it is in the code (processes remove it) or in the app and machine (they do not).
      Tasks 1–4 are shipped (the probe; the batch line's save CPU beside its wall time; the
      `file_hash` indexes — 0.085 s to add on the 6 GB database; WAL's safe flush setting and a 64 MB
      page cache — which also made the connection hook that enforces foreign keys actually run, as it
      never had); Tasks 5–12 (the batched writer, worker
      processes, workers that come back) are next — Claude.
- [ ] **A0 · WHERE THE INGEST'S TIME ACTUALLY GOES — measured at work, 2026-09-21.**
      From the app's own batch line: `load 0.0s | check 0.3s (62,323 new) | verify 0
      files 0.0s | process 674 files 520.1s`. **The pre-pass is free** — 0.3 s to
      classify 62,323 files — and every second is inside the per-file loop, at
      **771.7 ms/file**. `pool_probe` on the same machine puts parse+analyse through
      the SAME 4-thread pool at **339.2 ms/file**. So **56% of the ingest, 432 ms a
      file, is spent outside parsing** — and a process pool cannot touch any of it.
      By Amdahl that caps A3 alone at `1/(0.57 + 0.43/2.5)` = **1.35x**: 70 hours
      would become 52. **A3 is therefore NOT the first move; it was until this
      measurement.** The prime suspect is the save, which is serial by construction:
      `run_folder` pulls each result off the generator and calls `db.save_analysis`
      one at a time on the consuming thread while four parser threads contend for
      the same GIL — and save cost climbs as the database grows, with
      `cache_size` at SQLite's 2 MB default against a file heading past 5 GB.
      The batch line now prints `of which save Xs (N%, N ms/file)` and `rest`, so the
      next two-minute run says whether the save is the 432 ms or only part of it.
- [ ] **A3 · Processes instead of threads.** *Now gated on A0.* *After the rebuild.* Measured on 96
      real files: threads give **1.0×** at 1, 4, 8 or 16; processes give 3.6× at
      4 and **6.8× at 8**. The thread pool and its cap of 4 were built for the
      old 8 GB PC; the new laptop has 48 GB. Needs a design, not a patch:
      the processor writes "skipped" records to the database from inside the
      worker (six places); under tests each worker process would open the REAL
      database; and Windows starts processes differently from the Mac. No gain
      over the VPN (bandwidth-bound) — it pays on local files or a fast LAN.
- [ ] **A4 · Batch the saves** — **promoted above A3** by A0: saving is the SERIAL
      half of the loop, so it is the half a worker pool cannot help. The "~24 ms on
      the laptop" in the old note is stale — the speed probe measured 28.9 ms into an
      empty scratch database and **109 ms under the rebuild's own write load**, and
      the real database is far bigger than either. Pragmas are part of this
      (`synchronous=FULL` fsyncs every commit; `cache_size` is 2 MB). A/B'd on a copy
      of the 3.7 GB work database on the Mac: **3.9 ms/save either way** — a null
      result about THIS disk, not an answer about the laptop's, where a save costs
      7-28x more.
- [ ] **A5 · Worker count only ever goes down.** A memory warning drops a worker
      and nothing restores it for the rest of the run. Small; fold into A3.

## B. More useful information — the original goal

Spec: `docs/superpowers/specs/2026-09-17-process-recommendations-design.md`.
The app tells James what to change about the process to raise yield. It never
screens parts.

- [x] **B1 · Capture what the parser threw away** — every trim pass, the laser
      settings, and on laser 2 (DLTS) the per-position cut length, trim current,
      target output and measured output. 17 commits, `adcc1ce..5f94dc4`. Proven to
      change nothing that existed: 645 real files, 26 stored columns.
- [x] **B2 · Fresh-database rebuild — DONE** (James, finished 2026-09-22; carried home and
      verified 2026-09-23 with `scripts/verify_rebuild.py`: 6.23 GB, a fresh database, so all
      151,793 final-test records were graded by the current code). The copy home took three
      tries through OneDrive; `scripts/snapshot_db.py`
      now makes the one-file copy that avoids a torn journal.
- [x] **B2a · Working from home without the rebuild** (2026-09-20).
      `scripts/pull_model_slice.ps1` copies a few models' folders down the VPN
      (default 8232-1 and 8340-1, ~3 years); `scripts/build_dev_db.py` turns any
      folder into a throwaway development database with the new tables filled
      (refuses the real database by name). A 1,294-file sample database already
      exists on the Mac: `Work Files/dev_db/sample.db`, built in 69 s, 0 errors,
      839 captured trim passes. **8232-1 is the model to go deep on: 86% of its
      tracks need a second pass.**
- [x] **B3 · Re-derive every number that rests on a final-test verdict — DONE 2026-09-23**
      on the rebuilt database. **The resistance-target figures were laser-side numbers**
      (laser linearity verdict AND trimmed resistance inside the station's window), not
      final-test ones, so the re-grade could not move them — and they reproduce: **8232-1
      −0.63** over 30 months (best 2024-11, incoming 4,171 Ω → 73%; worst 2023-11, 4,550 Ω →
      10%), 8340-3 +0.54 (was +0.56), 6607 +0.35 (was +0.38); 6952, 8340-1 and 8202-1 still
      weak. **One changed: 8397-2 is now +0.25 (was −0.45)** — weak, and the other way.
      Against the re-graded FINAL TEST the link is weaker (8232-1 −0.34) — consistent with its
      failures being made at the laser and rescued by hand trim. Unchanged caveat: 8232-1's
      best months coincide with the one-cut → two-cut recipe change, which `ink_target`
      already holds constant. Nothing on the app's screens rests on a pre-rebuild final-test
      number; they compute live.
- [ ] **B4 · Four cheap questions about the trim passes, before any model.**
      *First look at question 1, 2026-09-20 — sample database, laser 2 (DLTS) only,
      482 tracks pooled across ~240 models, measured as error spread (max − min,
      immune to offset shifts):* **the first pass does NOT generally make
      linearity worse** — it did in only 10% of tracks; the median spread falls
      0.15 → 0.048. The one file seen on 09-17 (0.163 → 0.191 → 0.063) was the
      exception. What does stand out: tracks that end up needing 2+ passes START
      worse (0.174 vs 0.143) and the first pass leaves them at 0.065 where
      one-pass tracks reach 0.039. Thin and pooled — redo per model on the
      8232-1 slice. The questions:
      does the first pass make linearity worse (one file: 0.163 → 0.191 →
      0.063)? · which cut lengths pass first time? · was there a cut that
      avoids the second pass (2nd and 3rd passes are 23% of tracks)? · do the
      three lasers differ?
- [x] **B4 on 8232-1, first answers** (2026-09-20, home slice `Work Files/dev_db/slice.db`:
      9,862 files, 19,903 captured passes, 0 errors). Yardstick = best-offset worst
      point against the per-point limits; it matches the app's stored verdict on
      **2,472 of 2,472 tracks**, so it can be trusted on intermediate passes.
      - **"86% need a second pass" is a RECIPE CHANGE, not parts failing.** On laser 1
        (LTS) 8232-1 ran a one-cut recipe until mid-2024 (98% one cut in 2023 H2) and a
        two-cut recipe since (94–98% two cuts from 2025 H1). On laser 2 (DLTS, 2013–22)
        it was always two: 77% exactly two cuts, 15% three.
      - **The second cut is what buys the yield.** Same units, before and after: 18%
        inside limits after cut 1 → **52% after cut 2**. One-cut recipe ended at 23%
        trim PASS; two-cut at 53%. The first cut rarely makes things worse (4–8%).
      - **Incoming resistance still matters with the recipe held constant:** inside the
        two-cut era trim PASS fell 67% → 41% while median incoming R rose 4,136 → 4,660.
        Four half-year points, the first only 82 tracks — suggestive, not proven.
      - **So the 10% → 73% yield history credited to the resistance target has three
        changes tangled in it:** the move from laser 2 to laser 1 (2022–23), the recipe
        change (2024), and the target moves. Untangling them is B3 + finding #9.
      - Even with two cuts about half the tracks leave the laser outside the trim
        limits (median lands AT the limit, 0.99) — that gap is the cut-length model's target.
      - Pending: whether two cuts improve FINAL-TEST yield (needs the Test Station files).
- [x] **The company goal, in James's words (2026-09-20):** "the goal of the company is to not
      have to trim at all and if we do trim as little as possible to not overtrim and not
      tie up capacity at the laser." Hand trim is a fallback and does not guarantee a pass;
      **8232-1 and 8340-1 are hand-trimmed models** (ask for the full list). So the
      cut-length model's objective is **the least trim that lands in spec** — fewest
      passes, smallest cut — not FT yield on hand-trimmed models.
- [x] **How close is "no trim at all"? Graded every UNTRIMMED curve** (same validated yardstick;
      untrimmed sweep shares pass 1's position grid, 3,000/3,000):
      - 8232-1: linearity already inside limits before any cut on **2%** (laser 2, 4,032
        tracks) and **5%** (laser 1, 2,341). 8340-1: **1%** (2,320). Trimming on these
        models is driven by LINEARITY, not by resistance.
      - 8232-1's station settings: incoming 4,200–4,600 Ω, final 5,000–5,500 Ω — the
        windows do not overlap, so every unit must gain ≥ 400 Ω. But the linearity trim
        raises R by ~15–20% anyway (median +656 Ω on laser 1 vs +608 Ω needed to reach the
        floor), so the incoming target looks SET so the linearity trim lands R in spec.
        8340-1 is the opposite case: only 7% arrive below its 800 Ω floor.
      - **One file misled me:** "pass 1 under-cuts, pass 2 goes longer" is FALSE in general
        (under target after pass 1 on 54% of 3,861 tracks). 8232-1 ran a FIXED recipe on
        laser 2 for nine years — cut length 0.75 then 0.88 — and the extra cut in pass 2
        does not respond to where pass 1 landed (correlation +0.02). Nothing adapts to
        the part. That is the opening for a cut-length model.
      - New finding for the catalogue: **trim avoidance** — per model, the share of units
        arriving already in spec. Needs the full rebuild to rank every model.
- [x] **Two corrections from James, 2026-09-20.** (1) There is NO fixed list of hand-trim
      models: "if things fail we will make a determination of whether to try hand trim
      or not" — it is a per-failure decision; 8232-1 and 8340-1 are simply models where
      it happens often. The data can estimate how often per model (laser FAIL → final
      test PASS signature, finding #5). (2) `pred_deltas` / `used_deltas` are NOT "the
      machine's predicted vs actual correction": `pred_delta` is the ideal output step
      between neighbouring positions (+0.91 with the target ramp step, ~4,000 tracks),
      one value per pass; neither follows the resistance change (James's guess, tested:
      +0.01 / +0.07). The per-position data that matters is the cut applied plus the
      output before and after.
- [x] **B5 · Recommendation engine + the first three analyzers — SHIPPED** (2026-09-20, plan
      `docs/superpowers/plans/2026-09-20-process-findings-engine.md`, 11 tasks, every task reviewed).
      What you have: a **Findings** page in the sidebar (every model's findings, ranked by
      recoverable units a year), a **Findings tab** on the Model page (what was measured, then what
      to do about it, then the recipe history and the limit tables), a **Refresh process findings**
      button in Settings → Database, and a post-ingest phase that works them out by itself after
      every batch that saves trims. Three analyzers: ink target, recipe change, trim effort /
      trim avoidance — plus the limit-table analyzer from B6a below.
      On the rebuilt home slice (3 models): 8 findings in 17 s, no analyzer failures.
      It can say nothing, and when an analyzer CRASHES the screen names it instead of going quiet.
- [x] **B6a · Limit tables (catalogue finding #7) — SHIPPED** (2026-09-20, plan
      `docs/superpowers/plans/2026-09-20-limit-tables.md`). Found on your real database: **17
      (model, track, laser) groups are graded against two or more limit tables inside their latest
      12 months.** 8232-1 on laser 1 at 89 graded points until 2025 and 45 since (the 45-point
      table is the one final test uses); 8506A/B with every band loosened from ±0.01 V to ±0.0375 V
      in the first week of July 2025; 8340 with the later table wider at four end positions.
      A pass rate is a verdict against a test, so the analyzer reports these and never claims a
      gain — and `ink_target` now holds the limit table (and the station's final-resistance window)
      constant, because pooling across a change of test had it manufacturing an "aim lower"
      recommendation out of thin air.
- [x] **B6 · The findings catalogue — COMPLETE 2026-09-25** (plan
      `docs/superpowers/plans/2026-09-24-findings-catalogue-completion.md`; six analyzers, each reviewed,
      then a whole-branch review, ONE fix round and a re-review). On the home copy of the work database
      (327 models, refreshed): **176 findings** — yield 12, laser time 57, check 56, what changed 51 —
      and 0 models with analyzer errors. What the new ones found: **machine_compare** 2 (same model,
      same table, same months, two lasers, inside the model's own last two years); **loss_origin** 1
      (6607 laser 1: incoming linearity predicts the laser verdict); **station_setup** 33 (the laser
      and final test grade to different limits — D1's 8232-1 among them); **rework_load** 3 (6607,
      8340-1, 8232-1: laser FAILs that pass final test after hand trim, counted in unit-days, confirmed
      by a one-sided rank test against the same laser's untouched units nearest them — all three far
      below its p < 0.01 line);
      **ink_target** now says when the station's configured incoming window disagrees with the data;
      **setup_change** 39 on 16 models (B1c). Nothing claims a rate. The Findings tab shows the numbers
      behind each; facts cached by the older code read "Not worked out yet by this version" until the
      next refresh. Its review left small residuals for F5's round (below).
- [ ] ~~**B6 · The remaining eight findings**~~, one at a time, each with a test that it
      says nothing when there is nothing to say. The frame is built and proven; each is now a
      day's work. **#7 cut setting jumped the queue** (B6b below): it was the only one of the three
      not waiting on the rebuild — it needs trim data only — and it carries the largest measured
      effect in the data. Next two, in order: **#2 station setup mismatch** (the laser and final test
      grading to different limit tables — D1 below is its first case, and it needs the rebuilt
      final-test data) and **#5 rework load** (hand-trim volume per model).
- [x] **B4 questions 2, 3 and 4 — ANSWERED on the home slice** (2026-09-20, `slice_v2.db`).
      Yardstick = the exact best-offset test (an offset exists that puts every graded point in
      limits iff `max(lower−err) <= min(upper−err)`); it agrees with the app's own stored verdict
      on **1,484 of 1,484** laser-2 tracks, so it can be trusted on intermediate passes.
      - **Q2 · Which cut lengths pass first time? The per-pass cut setting is a STRONG lever —
        the biggest single effect found in this data so far.** Holding model, machine, limit table,
        incoming resistance and period all constant:

            cut setting   tracks   after cut 1   END OF LASER
        8232-1 · laser 2 (DLTS) · 2018-20 · one 45-point table
                   0.55       33          52%           97%
                   0.75      245          34%           78%
                   0.88      476           8%           38%
        8340-1 · laser 1 (LTS) · 2023-09 onward · one 53-point table
                   3000      414          19%           19%
                   3500    1,204          40%           41%   <- optimum
                   4000      408          23%           23%
                   4341       74           5%            5%
        8232-1 falls monotonically as the cut gets longer; 8340-1 has an **optimum at 3500**, and it
        wins in BOTH halves of incoming resistance (48% vs 23/27/7 below the median, 37% vs 19/13/5
        above). Median incoming resistance differs by under 60 Ω across 8340-1's four settings.
      - **The 0.88 setting was an 8-month EXPERIMENT, not the later recipe.** Dec 2018 → Aug 2019,
        inside a nine-year run of 0.75 (52 months at 0.75, 6 at 0.88, 1 month carrying both). Before
        it: 18% / 62%. During: 9% / 38%. After: 36% / 83%. *(Claude had this recorded as "0.75 then
        0.88" — wrong, and now corrected.)*
      - **Q3 · Was there a cut that avoids the second pass? The shop already ran the experiment and
        it went the WRONG way.** A longer cut did not remove the second pass — the median stayed at
        2 passes in every group — it just made both passes land out of limits. Shorter, not longer,
        is where first-pass success lives. This is the same direction as James's stated goal.
      - **Q4 · Do the lasers differ? They do not even store the same quantity.** `laser_cut_length`
        is 0.55–0.88 on laser 2 (DLTS) and 2500–4341 on laser 1 (LTS) — different units, never to be
        compared or pooled. And the lever does not act the same: 8232-1's three laser-1 settings
        (2950 / 4000 / 4100) sit at 43–51% end-of-laser with no separation, while 8340-1's four on
        the same machine span 5–41%.
      - **What this does NOT establish.** Settings are run in blocks: only 8% of 8340-1's 184
        production days and 2% of 8232-1's 86 used more than one setting, so this is a
        between-period comparison and a material-era effect cannot be ruled out from the data alone.
        The same-day pairs that exist are too thin to settle it (1–3 shared days per pair). **So this
        is a specific, cheap thing to TEST at the machine, not a number to act on blindly:** alternate
        3500 and 4000 on 8340-1 by lot for a week and the app will read the answer straight out.
- [x] **The per-POSITION cut column does not carry a model.** `cut_lengths` (laser 2's column 18)
      is nearly constant within a pass — 13.66 → 14.39 across a whole sweep, a ~5% range — and it
      explains nothing about what the sweep did: per-track correlation with the change in error at
      that position is **+0.02** (signed) and **+0.11** (absolute), and with the local step **+0.18**.
      The cumulative form correlates at |r|>0.5 on 50% of tracks — but so does cumulative *trim
      current*, in the same 50/50 split, which is the signature of "any monotone index tracks a
      wandering signal", not physics. Mean per-position cut is identical for tracks that passed on
      cut 1 and tracks that needed a second (14.132 vs 14.139). **Consequence for B7: the cut-length
      model's input is the per-PASS setting plus the incoming curve, not a per-position cut vector.**
      That is the second column in this family whose name promised more than it holds (after
      `pred_deltas` / `used_deltas`) — verify a column against data before believing its header.
- [x] **B6b · Cut setting (a new catalogue finding) — SHIPPED** (2026-09-20). The app now reads the
      B4 answer back out for EVERY model by itself, instead of it living in this file for two.
      It groups on (laser, track name, limit table) — never pooling machines whose cut numbers are
      not even the same quantity, never comparing two tests — finds what the laser is set to NOW,
      and reports a better setting only when it wins **in both halves of incoming resistance**, so
      the ink cannot be credited to the cut. It names the month a setting changed when the change
      was a clean switch, and always says how strong the evidence is: mixed on the same days, run
      side by side across months, or two separate periods.
      On the rebuilt home slice it finds **8232-1 on laser 1: the laser moved from 4100 to 4000 in
      July 2026 and has been 9 points worse since** (53% of 474 tracks vs 43% of 223, winning in
      both resistance halves) — and it stays SILENT on 8340-1, correctly, because the shop already
      moved to that model's best setting (3500) in January 2025. On the work database the engine ran
      over the five busiest models with no analyzer failure.
      Each silence rule was made to FAIL first: 8 mutations, 8 reds. Two tests passed for the wrong
      reason at first and were fixed — one fixture rendered a 35% pass rate as 40%, so the "too
      small to act on" test had a gain of zero and never reached the floor it was meant to exercise.
- [x] **B6c · Multi-pass burden (catalogue #6) — SHIPPED** (2026-09-20). Cuts the RECIPE did not
      ask for, and the laser time they cost — James's "not tie up capacity at the laser", as a
      number per model. The trap it exists to avoid is the one that caught the analysis by hand:
      "86% of 8232-1 needs a second pass" was a recipe change, not parts failing, so counting raw
      passes measures the process sheet. It works inside one laser, one track name and one cut
      setting, takes that group's OWN normal cut count, and reports only the work above it.
      On the home slice: 8232-1's two cuts are correctly silent (1.0% over recipe) while **8340-1
      takes more than its one cut on 28% of tracks — 40 unplanned laser passes per 100 tracks.**
      It claims no yield gain; freeing capacity is not a verdict change. 9 mutations, 9 reds —
      three tests passed for the wrong reason first (one asserted only the laser label, which
      survives pooling; one had no burden to suppress; one inserted the larger cut count first so
      the tie-break rule it meant to pin was never reached). A tie in "normal" now resolves to the
      LARGER count: deterministic, and a coin toss can never manufacture a burden.
- [x] **B1a · The `TrimVolts` sheets — CAPTURED (2026-09-24, `49f865e`, `b80c457`), back-fill yours.**
      Every NEW laser-1 `Trim N` pass stores its sheet as `increment_volts` (one curve per engaged
      position), `increment_volts_first_row` and `increment_volts_truncated`. Column *k* is the
      position at row (*start* + *k*) of `Trim N`, where *start* is `Points From Start` when the
      file names both `Points From Start` and `Points From End`, else `Initial Points Ignored` —
      **held against the machine's own `VOLTAGES` sheet it places 6,263 of 6,263 local passes
      exactly** (the first rule, the ignored counts alone, placed 6,227; the 36 misses were 8340-1
      files one position off). Passes already stored are filled by
      `scripts/backfill_increment_volts.py` at work (optional; `BRING_TO_WORK.md`, 2026-09-24, step 7), which
      checks each capture against `VOLTAGES` before writing and refuses one that disagrees.
      The history below is how it was found.
      **The original finding:**
      *(James, 2026-09-21: "i think it does in the trimvolts sheets?" — he was right and
      the spec, this tracker and CLAUDE.md all said laser 2 was the only machine with
      per-position data.)* **Verified:** 25 of 30 real LTS files carry `TrimVolts N`
      sheets; 0 of 30 DLTS files do; `TrimVolts` appears NOWHERE in `core/`, so the B1
      capture wave missed them entirely. **Layout** (measured on `lts_8232-1_193.xls`,
      49 positions × 25 rows): one COLUMN per trim position, one ROW per successive
      laser increment, each cell the output voltage after that increment. Live readings
      per position run 1-25 (mean 14.7) and climb steadily — so the sheet gives, per
      position, **how many increments were applied AND the material's response curve to
      them.** Real files span 49-120 columns and 14-103 rows.
      **Why this matters more than it sounds:** laser 2's per-position `cut_lengths` was
      captured and turned out to carry no signal (near-constant, corr +0.02). Laser 1's
      is the response curve itself, and **8232-1 has run on laser 1 since 2023** — so the
      model James most wants a cut-length model for has been producing the right data all
      along, into sheets nobody read. Needs: confirm the layout across models (column
      counts vary), decide the stored shape, then capture. Laser 3 unchecked.
      **Layout CONFIRMED and the design settled, 2026-09-23** (4,972 local laser-1 files,
      0 read errors, 32 models): `TrimVolts N` exists if and only if `Trim N` does (4,540
      files, no exception; the 432 without are 431 no-cut templates and 1 touch-up file).
      Column *k* is the position at row (*start* + *k*) of `Trim N` — *start* as above; the
      first version of this line said `Initial Points Ignored` alone, which r = 0.999998 could
      not tell apart from an off-by-one (reversed order refuted). Row 0 is read in every column — the only
      per-position pre-cut reading a laser-1 file has. Zeros are end padding only. **11
      sheets hit the old .xls 256-column limit and lose positions.** The last reading is NOT
      `Trim N`'s measured value (ratio 0.94–1.23): a live reading during the cut vs the
      verification sweep after it. `VOLTAGES`/`ERRORS` repeat what is already read;
      `TRIMDATA` is real but unexplained, so it is not captured. Design:
      `docs/superpowers/specs/2026-09-23-parse-fixes-design.md` §4; build: the parse-fixes
      plan, Tasks 4 (capture as `increment_volts` on each pass row) and 5 (a resumable
      back-fill for stored files, for James to run at work).
      **Laser 3 checked, 2026-09-23:** all 773 laser-3 (LTS3) passes carry cut length, trim
      current and used delta per position (99.8% live values in a 400-pass sample) — the same
      capture as laser 2, whose sheets laser 3 writes. Nothing more to read there.
- [x] **B1b · What the cut PATTERN says — investigated 2026-09-21** (James: "im more
      interested if the cut length or patterns can tell us anything… there is a lot of
      data available that might be able to help us"). Three answers, all on 8232-1's
      laser-2 history, which is the only place per-position data is captured today:
      - **The per-position CUT LENGTH is not adaptive.** It is near-uniform — positions
        given no cut at all are rare (median 0%), and the non-zero cuts spread only
        **4.1%** around their own median. Correlation of the cut at a position with the
        error the laser SAW going into that pass: **+0.027** signed, +0.072 magnitude
        (n=1,953 passes, each pass paired with the sweep that preceded it, not with the
        untrimmed curve — the first version of this test got that wrong).
      - **Because the adaptation is a control loop, not a varying command.** Each
        position has its own `trim_target`, and the laser cuts until it reaches it:
        **94% of positions land within 2 mV of target** (q1 83%, q3 98%).
      - **So the pattern worth having is WHERE IT FAILS TO REACH TARGET — and it is
        strongly directional.** Across 422,879 graded positions the miss rate climbs
        monotonically along the track: **13.7% at −20 → 15.6% at 0 → 17.6% at +20**, and
        38.4% at the far end (n=1,299). More than doubling end to end on the 5-unit
        buckets (8.4% → 19.6%). **Not yet attributable:** `laser_cut_direction` is
        recorded on laser 1 and is NULL on every laser-2 pass, and the miss can only be
        computed where target-vs-achieved exists — which today is laser 2 only. Testing
        whether cut direction explains it needs the `TrimVolts` capture (B1a). That is
        the single most interesting thread open.
- [x] **B1c · DONE 2026-09-25 — `setup_change`**: every captured setting on one (model, laser,
      track), segmented into stable setups (60 days / 100 tracks on each side, one limit table each),
      one finding per change naming every setting that differed, the pass rate stable setup against
      stable setup; per-unit readings (laser 2's `length_theoretical`, `starting_position`,
      `error_split_*` …) excluded by a curated list; the cut recipe left to `recipe_change`; a
      transition longer than 60 days kept as a fact. **The Response alias** (ruling below) is how it
      reads laser 1's `Response` and laser 2's `Response (Linear or Function)`: one setting.
- [ ] ~~**B1c · 46 setup parameters per file are CAPTURED and analysed by nothing.**~~
      `trim_setup.parameters` already holds laser power, duration (ns), pulse repetition
      rate, linearity velocity, inner/outer edge position, laser height, theoretical
      resistance, the error-split indexing setup and the resistance windows — since the
      B1 wave, for every file. **They vary enough to analyse:** 8232-1 on laser 2 shows
      12 distinct laser powers (52×2,899, 216×699, 72×316, 55×230), 12 pulse rates, 5
      durations; **8232-1 on laser 1 shows 8 laser powers** (50×750, 32×734, 36×595,
      70×271) and 8340-1 on laser 1 likewise. The `cut_setting` analyzer found a 9-point
      yield effect from ONE setting; its machinery (hold the limit table constant,
      require the winner in both halves of incoming resistance, grade the evidence,
      name the changeover month) generalises to each of these without redesign.
      **James, 2026-09-21: operators do not touch these; ENGINEERING can** — so they are
      a real lever, at an engineering lead time rather than same-day.
      Needs the fresh-database rebuild: the work database predates `trim_setup`. *(Rebuilt
      2026-09-22 — the data is there now.)* **Ruling (2026-09-23, parse-fixes spec §6):** laser 1's
      `Response` and laser 2's `Response (Linear or Function)` — probably one setting — stay split in
      storage; the setting sweep reads both under one name (a write-time alias would reach the 107,600
      stored rows only through a reprocess).
- [ ] **B7 · Cut-length model — PROMOTED** (James, 2026-09-20: "i want to do the
      cut length model i feel that is important"). No longer waits for the full
      rebuild: the home slice supplies real data now. Still gets its own design
      and approval, and still starts with the B4 questions — they are the cheap
      first step of this work, not a detour. Data for it:
      **Reshaped by the B4 answers above:** the input is the per-PASS cut setting and
      the incoming curve — the per-position cut column is nearly constant and carries no
      signal. The first question is no longer "what does a cut do at position i" but
      "what cut setting does THIS incoming curve want", and the data already shows the
      answer is model-specific (monotone on 8232-1, an optimum at 3500 on 8340-1).
      **laser 2 (DLTS) is the only machine recording the cut at every position.**
      8232-1's laser 2 history is 2013 → 2022-07 (4,178 files; it has run on
      laser 1 (LTS) since 2023-03). Models on laser 2 THIS year with heavy
      second-pass burden — where a recommendation could be acted on at the
      machine today: **7844 (51% need 2+ passes), 7845 (48%), 8762 (30%),
      6828 (27%)**; all of laser 2 is 28% over 11,199 tracks.

Traps the analyzers must respect (all written into the spec's known limits):
on laser 1 (LTS) the last captured pass duplicates the one before, so **pass
counts run one high** · `points_ignored_start/end` are always blank on laser 2 (DLTS)
and must never be read as 0 · `initial_trim_value` has its own column on passes saved
from 2026-09-24 on (G6) but lives inside `recipe` on every older one — read it through
`core/trim_passes.initial_trim_values(row, recipe)`, never the bare column · on a two-track
laser-2 file, track 2's resistance limits are `trim_setup.track2_parameters` (G7), and only on
files processed from 2026-09-24 on.

## C. Review and refactor

Agreed shape: **not** one big rewrite — the outputs go to customers. A review
that produces a ranked list by what each problem costs, then the worst items
one at a time, each proven against the 645-file baseline. *Starts after B2.*

- [x] **C1 · The review itself — DONE** (2026-09-20) → **`docs/CODE_REVIEW_2026-09-20.md`**,
      45 findings ranked by what each costs you, with what was fixed the same night and what needs
      a decision from you. Three reviewers read the whole codebase in parallel and every claim
      about data was checked against a read-only connection to the work database.
      Its one-line answer: **the app does not need a rewrite.** It needs three diseases treated —
      the same rule written in several places that have drifted apart, failures that are allowed
      to look like results, and safety nets that could not fail.
      Nine of the findings were fixed the same night (see "Done recently"); the rest are in §6 and
      §9 of that document, including the four that need your decision.
- [ ] **C2… · Refactors**, from the top of that list. *Not started — the sequencing above
      (after B2) is deliberate: the rebuild runs through this exact code.*
      **Two things measured 2026-09-20, before anyone starts step 1 ("delete the dead lines"):**
      - The honest count is **14 definitions / 669 lines** never named anywhere outside their own
        body, not 853. (A first pass said 96 defs / 5,837 lines — it used a regex that excluded
        matches preceded by a dot, which is exactly how every method is called. If a dead-code
        number ever looks too good, that is the bug to check for.) The 14: the whole unused
        alerts feature (`create_alert`, `get_unresolved_alerts`, `resolve_alert`,
        `acknowledge_alert`), `get_unmatched_ft_diagnostics`, `get_comparison_pairs`,
        `get_final_test`, `get_unprocessed_files`, `get_escape_scatter_data`, `get_trend_data`,
        `get_comparative_model_trends`, `get_cpk_trend_for_model`, `get_analysis`.
      - **`_set_sqlite_pragma` is in that list and MUST NOT be deleted.** It is a SQLAlchemy
        `@event.listens_for(self._engine, "connect")` listener (manager.py:259) — nothing names
        it because SQLAlchemy calls it. Deleting it turns **foreign-key enforcement off on every
        connection**, silently. Any "unreferenced" scan must skip decorated definitions.

- [ ] **C2 candidates parked by the parse-fixes reviews (2026-09-23/24)** — real, small, none urgent:
      - ~~`_update_existing_analysis` never calls `_record_processed_file`~~ — **FIXED 2026-09-25
        (`a6f2085`)**: a reprocess records what THIS run found.
      - ~~Three chart exports default a track's `linearity_pass` to True when it has no error data~~
        — **FIXED 2026-09-25 (`7b283c9`)**: such a track is "not graded" in all three (one shared helper).
      - Several `app_qa_sweep.py` check blocks have no try/except, so one exception ends the sweep
        instead of reporting one FAIL; the ERROR-reason check matches the marker row by exact path.
      - The TrimVolts back-fill re-selects passes whose file has no usable sheet on every run
        (`IS NULL` cannot tell "never tried" from "nothing there") — harmless, a little slow. *Left
        (2026-09-25): it runs once.*
      - `save_batch` does not write `trim_passes` (only `save_analysis` does) — fold into A4.
      - ~~**`get_database()` outside the app opens the production database read-write**~~ —
        **FIXED 2026-09-25 (`ffd2516`, `0250d95`)** after surveying every script: only the app's
        entry point allows the default; everything else names a database or is refused
        (`DefaultDatabaseRefused`, logged once per process). The parser-audit snapshot reads the
        default's `model_specs` read-only, so its numbers are what they were before the guard.
        Also closes code-review #27 (five scripts wrote to `data/analysis.db` by default).

## D. Checks at the shop — James

- [ ] **D1 · 8232-1: the laser and final test grade to different limit
      tables.** The only customer-facing mismatch of 36 found; James believes
      one station is set wrong. Same-day fix at the laser if so.
- [x] **D2 · Model 8706: 41 files skipped, only 2 parsed** — ANSWERED 2026-09-23 from the
      files themselves: **the parser is right.** All 47 skipped 8706 files are serial `0`,
      February 2016, many minutes apart on the same day — laser SETUP runs for a new model.
      They carry the parameter, notes and report sheets but no sweep sheet at all. The
      production variant 8706-3 has its `SEC1 TRK1` sheets and parses.
- [ ] **D3 · The QA sweep's one remaining red line — yours to run or leave.** On the rebuild
      (full sweep on a copy, 2026-09-24: 291 checks, 1 FAIL, 8 WARN) it is *trim/FT link points at
      the day's FINAL trim attempt*: **4 final-test links** point at an earlier trim attempt of
      the same unit on the same day. The remedy exists — `.\.venv\Scripts\python
      scripts\repair_trim_ft_links.py data\analysis.db` (it re-reads each trim file's clock time
      from its name, then re-points EVERY final-test link — a full rematch, so a long run; it asks
      before writing; snapshot with `scripts\snapshot_db.py` first) — and until it runs,
      escapes/overkills on those units are close but not exact. Four links: not urgent.
- [ ] **D4 · Laser 1's old template: which starting point does `TrimVolts` follow?** The older
      template of 6607 and 8232-1 labels the slot `Start Point` / `End point` ("points from start
      for reading/measuring" — not trimming). It equals `Initial Points Ignored` on every local
      file, so no workbook here can say which the machine follows when they differ — and they do on
      1,223 old (2011–2016) files on the work database, holding 14 stored `Trim N` passes. The
      capture uses the ignored counts there. **Nothing to do unless the back-fill (B1a) lists one
      of those files under "placement disagrees"** — it checks every curve against the file's own
      `VOLTAGES` sheet and refuses one that does not match, so a wrong rule cannot land silently.
- [ ] **D5 · What is "Micro-Lin max. error slope" for?** A setting/check on about 1% of laser-2
      (DLTS) files, found in the 2026-09-24 survey of what those files carry. Not read today. If it
      is a real acceptance check at the machine, it is worth capturing; if it is a leftover, not.
- [ ] **D7 · What screen resolution and Windows display scaling does the work laptop use?** The app
      lets its window shrink to 960×640, and below 1280×720 some text is cut (the audit counts 261
      clipped widgets at 960×640, on every page). Raising the minimum to 1280×720 would stop that —
      but at 150% scaling (a common laptop default) 1280×720 is the whole screen, so it waits on
      your answer. At your saved size and at 1280×720 nothing is cut.
- [ ] **D6 · 8506A / 8506B on laser 2: were the limits loosened by ECN?** Every band went from
      ±0.01 V to ±0.0375 V in the first week of July 2025 (pass rate 83 → 100 %, 70 → 100 %). The
      limit-table analyzer reports it as a change of TEST, never as a yield gain.
- [ ] **D8 · 8397-2: which element does a "Section 3_0 Test" final-test file test?** Its serials are
      digits only, so the rework-load analyzer (next pull) pairs each one with the unit's one
      trimmed track. If Section 3 is a different element, that pairing is wrong. It changes nothing
      today (only 5 comparable untouched units a year, too few to judge), but it will once there are more.

## E. Settings: one backlog upload instead of two inputs

James, 2026-09-20: Settings has separate "Active models" and "Pricing" inputs; "we should
simplify this and allow me to upload a current backlog to get active models and pricing
all in one." A backlog export is in `Work Files/` (gitignored — it holds customer names,
PO numbers and prices, and must never be committed).

- [x] **E1 · SHIPPED 2026-09-20 (`90f66f0`)** — one upload in Settings sets the active models
      (replaced on each upload) and the prices (merged — decision 7 above). The design, as built:
      What a first look at the file showed: one sheet, one row
      per open order line, with `Item ID`, `Balance`, `Unit Price`, `Need Date`. 178 distinct
      items have an open balance; **98 are models the app knows, and they hold 95% of the
      open units**. The other 80 are mostly variants (`… FAI`, `… LAT`, `… TEST UNITS`) or
      products the app never sees. 31 of the 98 carry more than one unit price, so a price
      rule is needed. The hand-pinned active list is EMPTY today — nobody maintains it,
      which is the case for replacing it. The existing price importer already recognises
      this file's `Item ID` / `Unit Price` columns. Store only model, open quantity, price
      and need date — never customer or PO. Possible bonus: rank Findings by real open
      demand instead of last year's track count. **DESIGN SETTLED by James, 2026-09-20:**
      1. **Price = the LATEST order's unit price** for that model ("that's the current
         price") — latest by `Order Date`, not the most common and not an average.
      2. **`FAI`, `LAT`, `TEST UNITS` and the like are add-ons and are NOT included** —
         neither as models nor folded into a base model. Only an `Item ID` that exactly
         matches a model the app knows counts.
      3. **Each upload REPLACES the active list.** Manual pinning stays for exceptions.
      Queued behind the engine build.

## F. The facelift — James, 2026-09-23

"i think the app needs a facelift or polish, i dont like the design, lastly the finings sheet
i dont like the layout its just a bunch of rows and its hard to see whats important."

- [x] **F1 · Design, step 1 — APPROVED section by section.**
      `docs/superpowers/specs/2026-09-23-design-system-and-findings-page-design.md`, with the
      approved picture beside it (`…-findings-page-mockup.html`). James chose: Findings
      **grouped by issue** (four groups); what is wrong today is **dated or generic** and
      **hard to read**; direction **refined dark** (today's navy, brighter text, bigger
      numbers, teal as the one accent); scope **restyle AND relayout every page**, delivered
      in steps. Measured while designing: three of today's colour pairs fail the basic
      readability minimum, including the out-of-control drift tier; the app has no colour for
      PASS or FAIL at all; only 9 of 96 findings claim a gain and all 96 are drawn the same.
- [x] **F2 · Build step 1 — SHIPPED** (`28a4862..4338b01`, 24 commits). New refined-dark
      palette and a readability test; shared building blocks, sentence-case headers and a
      page caption; chart text on the theme's scale with laser colours kept clear of
      pass/fail meaning; the two `cut_setting` fixes (a short trial cannot be "best", "now
      running" means now — retires 8397-2's "+48 points"); the Findings page's rules as
      tested data; the Findings page rebuilt as four groups with one open row and a
      merged-track table; the Model page's Findings tab on that same view;
      `scripts/render_pages.py` with a mechanical `--audit` for clipped text, plus the 14
      overflow fixes it found. **Final-review fix round (2026-09-24, `4443bbf..8631b34`):**
      the selected tab/segment readable (new `SEGMENT_SELECTED`, 4.8:1; a widget-level
      contrast test that reads what the controls really draw) and dark checkmarks; every
      Findings row opens its own evidence (a key per track/table); a pass-rate move across a
      limit-table change is uncoloured and tagged "different test"; cut_setting says when only
      a TABLE went quiet and never merges two tables; the audit sees squeezed-out and
      past-the-edge text, the real settings and the tab names, and writes before teardown;
      plus the minors (local caption date, Home wraps and notices, four shouting strings).
      Its one open line: at 1280×720 the Triage model list gets no room (left for Triage's step).
      **Task 4, bundled IBM Plex fonts — DONE** (James said yes in chat, 2026-09-24). Six
      files (~0.9 MB, four TTFs plus IBM's `license.txt` and Google's `OFL.txt`, both SIL OFL
      1.1) live in `gui/v6/fonts/`. `font_loader.py` loads them privately for Tk on Windows
      (CustomTkinter's `windows_load_font(..., enumerable=True)` — its own default hides the
      family from `tkinter.font.families()`) and into matplotlib everywhere; a missing or
      failed file logs a warning and stays on the fallback, never raises. `theme.py`'s
      `FONT_FAMILY_MEDIUM` now carries two spellings, because the bundled Medium file's real
      legacy family (read with fontTools) is the abbreviated `"IBM Plex Sans Medm"`, not
      `"IBM Plex Sans Medium"`. `tests/test_font_loader.py` (9 tests) pins file/licence
      presence, the family match against the actual files, the graceful-fallback path, and the
      Windows call itself (faked — this Mac can't take that branch for real). See
      `BRING_TO_WORK.md` for what James will see and how to confirm it at the work machine.
- [x] **F3 · The other six pages — DONE 2026-09-24** (plan `docs/superpowers/plans/2026-09-24-facelift-step2-pages.md`,
      every task reviewed; then a whole-branch review, ONE fix round (9 commits, `f194a2b..e415fcd`) and a
      re-review, 2026-09-25; `render_pages.py --audit --scaling 1.0/1.25/1.5`: 0 clipped at the saved
      size and 1280×720 at each — wrapped text used to be scaled twice, so above 100% Windows scaling
      every wrapped line was cut; every failed load is named, never "0" or "nothing"; one teal button
      on the model page whichever tab is open).
      Each page now leads with its most important thing, in words: **Investigate** — the verdict
      in the caption, "Worth changing on this model" first, then "How it's running"; **Home** —
      N worth changing · M drifting now; **Settings** — Ingest folders first, sentence case, no
      teal; **Dashboard** — every failed query named, never drawn as zero; **Triage** — "Needs a
      look" / "All models", fits at 1280×720; **Process** — one button, "See what changed".
      **The Units chart was redrawn after James said it "looks horrible"** (2026-09-24): faint units,
      one 30-day median line (90-day past 18 months), red = beyond ±3σ, ▲ = months off the chart —
      prototyped on real data first, then built; it shows the window you pick (90 days by default;
      "All" really is all since the fix round). Was: *Relayout each remaining page*, one design round each: Model, Home,
      Dashboard, Triage, Process, Settings. **Designed and planned 2026-09-24** (rulings, yours to
      overturn once you have seen the pages): `docs/superpowers/specs/2026-09-24-facelift-step2-pages-design.md`,
      plan `docs/superpowers/plans/2026-09-24-facelift-step2-pages.md`. Next to build.

- [x] **F5 · DONE 2026-09-25** — 196 of 326 models read Inactive on the home copy (190 by date, 6 "no trims
      on record": every file they have is a no-cut check sweep); nothing hidden, counts unchanged, the
      top-three previews put active models first. With it: F4's review gaps and the findings
      catalogue's re-review residuals below, and a callback that keeps failing is logged once, then
      counted. **A question for you:** a model pinned in Settings → Active Models (the backlog) can
      also read Inactive (not trimmed in two years) — say if either word should change.
      Was: **F5 · "Inactive" models — James, 2026-09-25:** *"models that havnt been trimmed in 2 years should
      show inacative or something but i dont want to hide them as the data is useful if we decide to
      start building the model again."* A model whose newest trim file is more than two years older than
      the newest file in the database is labelled **Inactive · last trimmed Mon YYYY** wherever it appears
      (findings rows, the model page caption, the Triage list); nothing is hidden and every finding and
      number is kept; the short ranked previews (Home's top three) list active models first. 40 of the
      240 findings on the rebuild are on such models today. Claude, next. **With it** (the findings
      catalogue's re-review residuals): one "newest trim file" helper with the engine's future-date
      guard (machine_compare's per-model window has none); the 60-day cap compares whole timedeltas
      (60 d 20 h counts as within today — one finding); the failed-predictor line shows even without a
      loss section; history moves under a point read "±0", not "-0".
- [x] **F4 · Step 2 follow-ups — DONE 2026-09-25** (`dbf57d0..7ab3a67`, 9 commits, reviewed:
      approved; gate 123/123 files, 2,436 tests; audit 0 clipped at 100/125/150%, twice at 150%;
      sweep 299 checks, 1 FAIL = D3). Its review left robustness gaps for **F5's round**: the redraw
      guard has no counting test; the private CustomTkinter override is not named by the version
      pin's message; one crash in the sweep's section 5 shows as three FAILs; Home's FOCUS apply now
      stops the two ingest notices if it raises; the Findings page and the Dashboard's `reload_now`
      lack the stale-load counter; `safe_after` swallows callback errors without logging them.
      Found by the fix round's re-review (2026-09-25); all pre-existing, none blocking:
      - **A scrolling tab comes up blank after you click away and back** (Findings, Drift metrics,
        Smoothness) — seen on the Mac at every scaling; probably not on Windows, unverified.
        Resizing the window brings it back. Fix: nudge the tab's scroll frames when it is shown.
      - A CustomTkinter race: a click on a tab within 100 ms of the app switching tabs itself leaves
        the tab area empty (reachable only right after a findings link).
      - The Findings tab on a failed facts load says "not computed yet" under the banner that names
        the crash — the three-state rule, on the tab itself.
      - Home and Triage can let an older, slower load overwrite a newer one (the Model page guards
        against this; they do not).
      - The Units view's 12-month framing never trims anything now (the page window defaults to 90
        days) — dead code that one test and one chart-QA section still rest on.

## G. Parse fixes — found 2026-09-23, BUILT 2026-09-23/24

Design (rulings, James's to overturn): `docs/superpowers/specs/2026-09-23-parse-fixes-design.md`.
Plan: `docs/superpowers/plans/2026-09-23-parse-fixes.md`. Every item below was made to fail
first and reviewed; the whole test suite is the gate.

- [x] **G1 · A track that failed processing was counted as a linearity FAIL — FIXED** (`b4f640f`,
      `97a8dd4`, `e04f93e`): it carries no verdict now, and one stored the old way reads back with
      none, whatever is stored — no reprocess needed. A track with
      fewer than 10 points is stored ERROR *and* `linearity_pass = 0`, and four Findings
      analyzers count every non-empty verdict. **8856 on laser 2 (DLTS) reads 27.7% when its
      graded tracks pass 49.0%; 8856-1 57.0% vs 72.1%.** Plan Task 1.
- [x] **G2 · V5's Settings → Apply ML would rewrite verdicts.** Its bulk update grades
      status as "both pass → PASS, both fail → FAIL, else WARNING": one click would turn up
      to 1,701 linearity FAILs into WARNING, take all 235 ERRORs out of ERROR and make 6,973
      untrimmed sweeps WARNING. **Not run on the rebuild** (checked). Only V5 has the button.
      Plan Task 2. **FIXED** (`3bc371c`): Apply moves sigma, never the linearity verdict, and
      leaves ERROR and UNTRIMMED alone.
- [x] **G3 · No ERROR says why — FIXED** (`96e2dba`, `c3b9f33`). None of the 237 on the rebuild carries a reason the app can
      show; the words exist on the tracks (140 bad limit columns, 94 too few points) or on a
      separate marker row (3). Plan Task 3 stores one reason and shows it on the Model page — the
      unit list's empty linearity cell reads "not graded: <reason>"; 234 of the 237 explained
      today, the other 3 when next reprocessed.
- [x] **G4 · Laser 1's `TrimVolts` capture** (B1a) and its back-fill. Plan Tasks 4–5 — BUILT;
      the back-fill is yours to run at work (optional).
- [x] **G5 · The database path does not travel — and two small ones. FIXED** (`d391f6a`,
      on `main` since the morning of 2026-09-24): the path is saved relative to the app folder, and one written by the other
      operating system falls back to the default with a warning. `config.yaml` stores
      the path absolute (`C:\dev\…\data\analysis.db`). **On the Mac that string is a relative
      FILE NAME, so the app opens a junk database in the repo folder instead of the rebuilt
      one** (checked 2026-09-23: it would open `C:\dev\…\analysis.db` relative to the current
      folder). Workaround until the fix: set `database: path:` in `data/config.yaml` to
      `data/analysis.db`. The junk file in the Mac repo root holds 1 processed file and 6
      final-test rows written by a local test run, nothing of yours — delete it when
      convenient. Also: `Unknown` (381 test files) is returned as a known model; floats are
      written into two integer columns (measured impact zero). Plan Task 6.
- [x] **G6 · The initial trim value per position gets its own column** (`744df1a`, found by the
      2026-09-24 survey of what laser 2 / laser 3 files carry): laser 2's pass sheets record, per
      position, the trim target, the INITIAL trim value and the final one; the initial was being
      stored inside the pass's `recipe` blob. New passes store it in `initial_trim_value`; the
      ~83,000 existing laser-2/3 passes keep it in `recipe` (no start-up rewrite) and one helper
      reads both the same way (83,264 of 83,488 carry a value; none was lost in the move).
- [x] **G7 · Track 2's own setup on two-track laser-2 files** (`9bb8e40`, same survey): the
      `Track Parameters` sheet carries TRACK 2's whole setup in column C, and it was dropped, so
      Findings judged a track-2 track against TRACK 1's resistance limits. Stored now as
      `trim_setup.track2_parameters` (112 local files, 57 models), and Findings uses it.
      **Not back-filled: the 1,868 two-track analyses already in the database are judged against
      track 1's limits until they are next processed.** Laser 3 (LTS3): 0 of its 547 analyses
      are two-track.
- [x] **G9 · What the final whole-branch review found, fixed** (`9d10803`..`a46b5dc`): a start-up
      backfill that cannot write no longer stops the app opening; **ingest now checks each laser-1
      TrimVolts capture against the workbook's own `VOLTAGES` sheet and refuses one it contradicts**
      (the back-fill's rule, one function for both — a refused pass keeps its sweep, stores no curves,
      and the log names it); the Units tab's "not graded" and sigma dash follow the TRACK, not the
      file; track 2's limits replace the file's as pairs; two sweep checks that would have gone red
      at work on correct data (or passed over nothing) are fixed; V5's Apply stamps its threshold
      only on the tracks it grades. Parse gate: nothing moved.
- [x] **G8 · The customer-value guard cried wolf** (`595d49c`): a real 5-digit PO number
      happened to be the last digits of a long measured decimal in a test baseline. A short PO
      now matches only as a whole number; the whole history stays clean.

## Housekeeping

- [x] **H1 · `CLAUDE.md` step 1 breaks the git remote — FIXED (`fce84ee`):** the step now runs
      only when `.env` exists and holds a token. There is no `.env`, so
      the command sets the remote to `https://@github.com/…` and the next push
      fails. The plain URL plus the keychain works. *James's call to delete it.*
- [x] **H2 · Leftovers from July — DONE (`ede629e`):** the script is kept, its stale renders
      are ignored. Were:
      `docs/chart_rep_review_2026-07-16/` and `scripts/chart_rep_review.py`.
      Keep, commit or delete?
- [ ] **H3 · The decision logs: READ (2026-09-20), do NOT delete yet.** *(2026-09-24: the ten
      citing lines below now point at `docs/decisions/2026-09-ledger-decisions.md`, which carries
      what each cited note said — so deleting the workspaces no longer breaks a tracked file. It
      stays your call; nothing needs them deleted.)* All five ledgers were
      read and every judgement call checked against the code, the tests and this file: **130 calls,
      56 already written down somewhere durable, 51 pure process, 23 that would have been lost.**
      Two things came out of it that change the job:
      - **Deleting the workspaces breaks ten tracked files that cite them** — `utils/threads.py`,
        `tests/conftest.py`, `tests/test_tk_thread_safety.py`, `tests/test_model_page_failures.py`,
        `config.py`, `gui/v6/sections/backlog.py`, `tests/test_backlog_section.py`,
        `tests/test_trim_capture_db.py`, `TRACKER.md` itself — and
        `docs/superpowers/plans/2026-09-20-limit-tables.md` says "the patches ARE the code", so
        deleting them makes that plan unexecutable. **So: inline the cited substance first, or keep
        `H5-report.md`, `M1-brief.md`, `task-E1-brief.md`, `task-7-brief.md` and the limit-tables
        patches.** Not a five-second delete.
      - **One latent defect that nobody had carried forward** — see the next line; it was the most
        dangerous of the 23 and is now fixed.
      **Two of the 23 are now closed** (see "Done recently"): the `pass_index=None` hole, which
      could have rolled back a whole file's analysis save mid-rebuild, and the duplicated
      pass-sheet regex, now pinned by a test instead of merged.
      The other 20 are small (a `Response` field that may be the same field on two machines and is
      now split; two differently-scoped pass
      rates on one screen from `limit_tables`; a float stored into an Integer column). They are
      written up in the ledgers, which is why the ledgers stay until their substance is inlined.
- [x] **A blank THEORY cell can no longer become an error of 0.0** (2026-09-20, found while
      reading the ledgers). `final_test_parser.py` honours "blank is ungraded, never 0.0" in the
      branch that uses the file's own error column — and the branch 20 lines below substituted the
      MEASURED value for a blank theory cell, making `measured − measured` exactly 0.0: dead centre
      of every band, on a zero-tolerance metric. The same branch wrote 0.0 for every point of a
      track with fewer than two points, handing it a flawless sweep. **Measured impact: zero** —
      608 real final-test tracks parsed before and after, 0 changed, 0 verdict flips, and the
      645-file gate is unchanged. It is LATENT, not active; it is fixed because the rebuild runs
      this code over ~151,000 final-test records rather than 608. Both branches are pinned by tests
      that go red when reverted.
- [ ] **H4 · Retire V5?** It holds nothing V6 lacks; it stays until James says.

## Parked — real, not scheduled

- 35 models have a position column that does not run in order in stored
  data — a suspected ingest fault, its own investigation.
- The model specifications table is empty in production; resistance limits
  come from the laser files and the ATP documents.
- ATP audit: 36 real differences, 29 of them trim-only
  (`scripts/atp_spec_audit.py`).

## Done recently

**2026-09-24 night:** facelift step 2 (F3) — the other six pages, and the Units chart redrawn after
James's "that chart looks horrible"; three streams ran at once in separate worktrees after James
asked why only one task ran at a time (pages, findings analyzers, refactors). C2 step 1: 13 unused
`DatabaseManager` definitions deleted (683 lines), each re-proved unused by a separate reviewer.

**2026-09-23/24** (every item made to fail first and reviewed; the whole suite green at each step):

- **Facelift step 1 finished** (F2 + fonts): the dark theme, readable text, the Findings page grouped
  by issue, and IBM Plex bundled — loaded privately on Windows, no install; charts use it everywhere.
- **Parse fixes G1–G8** (section G): a failed track is never a FAIL, V5's Apply keeps verdicts, every
  ERROR says why, laser 1's TrimVolts captured with a back-fill ready, the database path travels,
  the initial trim value and track 2's own setup stored, the customer guard no longer cries wolf.
- **B3 re-derived on the rebuild**, **D2 answered** (8706's skipped files are laser setup runs).
- **The home database was written to by two of my scripts** — schema and bookkeeping only, proven;
  the file is read-only on the Mac now. See the top of this file.

**The night of 2026-09-20** (one session; every item tested, each fix made to fail first):

- **Four fixes to what gets STORED**, all described at the top of this file: final tests the station
  failed being stored as passes; units that were never cut being stored as flawless; a thread race
  in the morning's speed fix that could record a good file as an ERROR; `999.999` markers being
  averaged into the evidence workbook and the drift detector.
- **The findings engine and the limit-table analyzer** (B5, B6a above).
- **One backlog upload** replaces the active-models and pricing inputs in Settings.
- **The test suite runs**: it never took 2.5 hours, it hung. ~1,800 tests in about 3 minutes;
  `python scripts/run_test_gate.py` is the gate now.
- **The 645-file parse gate compares every value**, not just whether a file parsed. It had been
  described as a full diff in two places, including by me.
- **One production-database guard** (`scripts/_db_guard.py`) compares the FILE, not the path text,
  and no test names the real database any more.
- **Screens stop lying when a query fails**: the Model page names what could not be loaded instead
  of rendering "no data", never leaves the previous model's verdict on screen, and no longer says
  "NOT TRAINED" when the drift query simply failed.
- Claude's own mistake, fixed: an example price in a test was a real one from the backlog export.
  Scrubbed from local history before the push; **`python scripts/check_no_customer_values.py`**
  now scans every version of every file in a commit range, plus every commit message, for a real
  price beside its model, a customer name or a PO number — and prints only WHERE, never the value.
  Proven on a throwaway repository: it passes invented data and fails a real price and a real
  customer name.

- **The customer-value guard had two defects of its own, and now has tests.** Scanning the whole
  history turned up one report — against a commit from April that had been on `main` for months.
  It was a FALSE POSITIVE: model 8035-2 has a zero-priced backlog line, and `0.00` matched the "00"
  of a `9-00 AM` timestamp inside a data filename. Fixed twice over (a zero unit price is not
  customer information; a number inside a path, filename or date is not a price) — each fix alone
  clears it, measured. `tests/test_customer_value_guard.py` now pins it: 14 tests, and all four
  rules were mutated to prove they bite. Two of those tests passed for the wrong reason when first
  written and were rebuilt. **Nothing leaked; the whole history scans clean.**

- **Two latent faults closed before the rebuild, both found by reading the build ledgers.**
  A trim pass with no `pass_index` would have hit `nullable=False` at flush and rolled back the
  WHOLE analysis save — every track's verdict for that file — to save one pass row; the guard
  written to prevent exactly that did not catch it, because `None` is a perfectly good set member.
  And the two copies of the pass-sheet regex (`parser.py`, `trim_passes.py`) that decide which
  sheets are trim passes are now pinned by a test, so they cannot answer differently about the
  same workbook. Both proven by reverting the fix and watching the test go red.

- 2026-09-20 · the app now SHOWS the shop's laser names — "Laser 1 (LTS)",
  "Laser 2 (DLTS)", "Laser 3 (LTS3)" — on the company trend chart (legend in
  shop order), the Excel and unit-chart exports, and the V5 classic screens.
  Stored letters and parsing logic untouched; one function, `laser_label()`.

- 2026-09-20 · speed: one read and one lookup per file; skip warning no longer
  blames laser 3; LTS3 confirmed long since validated (547 files, 0 errors)
- 2026-09-18 · capture build shipped; disk estimate corrected 200 MB → 1.6 GB
- 2026-09-17 · repeat files stop repeating; ATP audit corrected 248 → 36
