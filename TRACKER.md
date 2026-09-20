# Tracker

What is open, in what order, and who holds the next move. Updated in the same
commit as the work it describes. `BRING_TO_WORK.md` stays the place for
step-by-step instructions at the work machine; this is the index above it.

Last updated: 2026-09-20

## ▶ Read this before you start the rebuild (Claude, night of 2026-09-20)

**Pushed to `main` on the night of 2026-09-20** (commit `22a91dc`, 50 commits). At work:

    git pull

Then follow the rebuild checklist in `BRING_TO_WORK.md` as written. Optional, ~4 minutes, worth it
once: `.venv\Scripts\python scripts\run_test_gate.py` — the whole test suite, every file OK.

### Four fixes that change what the rebuild stores — take these with you

| | What was wrong | Evidence it is right now |
|---|---|---|
| 1 | **Final tests the station FAILED were stored as PASS.** When a sheet's error column is under 90 % filled (normal when the station ignores more than 10 % of its rows) the app recomputed the errors, divided them by full scale, and graded them against limits still in volts. | Real pipeline, before and after: 3,766 of 3,768 slice tracks unchanged in all 13 fields; of the 12 affected tracks 6 flip, **all PASS → FAIL**; agreement with the station's own verdict **5/12 → 11/12**; no station-PASS became a FAIL. ≈1.3 % of final-test files. |
| 2 | **Units that were never cut were stored as flawless.** When no cut is made, laser 1 still writes a `Lin Error` sheet — the blank template. The app took it for the final sweep: zero error, linearity PASS. **1,182 tracks** (6126: 455 · 6607: 224 · 8340: 193 · 8232-1: 163 · 1844205: 92). | In your slice the workbook's REAL sweep is out of limits in **111 of 130**. They now load as UNTRIMMED, and no fake laser pass is written for them. Rebuilt slice: 8232-1 laser-1 yield **42.0 % → 39.1 %**; 0 errors in 13,031 files. |
| 3 | **A race in my own speed fix from this morning** (already on `main`): the ingest workers share two caches with no lock. A collision records a GOOD file as ERROR. | Reproduced (325 exceptions in 6 s under a forced-switch loop), fixed, and a test proves the slow network read still happens outside the lock. |
| 4 | `999.999` error markers were averaged into the evidence workbook and fed to the drift detector (8856, 2024-07: "mean sigma" **432.99** against a true 0.0012). | One shared definition, both consumers, tests red before and green after. |

**After the rebuild expect:** laser yield a little LOWER on models with no-cut files (that is the fake
passes leaving), a few hundred final-test verdicts moving PASS → FAIL, and ~1,100 more UNTRIMMED records.

### What is new

- **Findings** — a sidebar page ranking every model's findings, a Findings tab on the Model page, a
  "Refresh process findings" button in Settings → Database. It fills in by itself after each ingest.
  Three analyzers so far: ink target (held to one laser, one recipe **and one limit table**), recipe
  change, trim effort / trim avoidance. It says nothing when there is nothing to say, and names an
  analyzer that crashed instead of going quiet.
- **One backlog upload** in Settings replaces the active-model and pricing inputs (latest order's price;
  FAI / LAT / test-unit lines never count; the active list is replaced, prices are merged).
- **The test suite runs.** It never took 2.5 hours — it HUNG (a test fixture kept a second Tk window
  alive). **1,7xx tests, 0 failures, ~3 minutes.** New gate: `python scripts/run_test_gate.py`.
- **The whole-app review you asked for:** `docs/CODE_REVIEW_2026-09-20.md` — 45 findings, ranked by
  what they cost you, with what was fixed tonight and what needs your decision.

### Something you may not know: 17 model/laser combinations are graded against TWO limit tables

Within their latest 12 months (a table counted only with 30+ tracks; per track name):
**8232-1 on laser 1** — 89 graded points until 2025, 45 since, same band; the 45-point table is the one
final test uses (34 % vs 49 % leave the laser inside limits). **8506A / 8506B on laser 2** — every band
loosened from ±0.01 V to ±0.0375 V in the first week of July 2025 (83 → 100 %, 70 → 100 %).
**8340 on laser 1** — the later table is wider at four end positions by up to 0.164 V (34 → 82 %).
A pass rate is a verdict against a test; every trend across such a change compares two tests. The app
now reports these as findings (lever: laser limit table, same day). **Were the 8506 limits changed by
ECN? Which 8232-1 table is the intended one?**

### Decisions that are yours (none blocks the rebuild)

1. **Ungraded final tests count as PASS** at file level and FAIL at track level. Small overall (0.19 %),
   large per model: 8502 reads 27.6 % FT pass, 15.5 % over files that were really graded.
2. **One definition of yield.** The Excel export says 12.3 %, the app says 76.0 %, same data (the
   export counts WARNING as a failure; your rule says sigma is never a rejection).
3. **"Fix Missing Tracks" writes invented numbers** (sigma 0, spec 0.02). Proposal: write blanks.
4. **Five backfill scripts write to the work database by default**; one has no dry run at all.
5. **A full final-test rematch can fire at app startup** after a migration — the shape of the 09-14 night.
6. Sidebar: **Findings** sits third, after Investigate. Say if you would rather it lived on Home.
7. Backlog prices are MERGED on upload (a model that drops off the backlog keeps its last price).
8. A no-cut check sweep is a real measurement of the unit at that moment. Should it count for anything?

### One thing of mine to own

A test I briefed used a real (model, price) pair from your backlog as "example data". It was committed
locally, never pushed; I rewrote the local history to remove it before pushing. `scripts/check_no_customer_values.py`
now scans a commit range — every version of every file, and every commit message — for a real price
beside its model, a customer name or a PO number, and prints only where, never the value. Local
commit hashes changed as a result of the rewrite.

## Which laser is which

The shop's numbers do NOT follow the code's letters (James, 2026-09-20):

| Shop name | Folder | Code | Records |
|---|---|---|---|
| **Laser 1** | `LTS` | System B | each pass's sweep + that pass's settings |
| **Laser 2** | `DLTS` | System A | the same, **plus the cut applied, trim current, target output and measured output at every position** |
| **Laser 3** | `LTS3` | System C | same format as laser 1 |

## The critical path — read this first

Three workstreams are open, and almost everything in all three waits on **one
overnight job**, which waits on **one 60-second measurement**:

```
 1. Probe at work  ──►  2. Pick where the   ──►  3. Fresh-database  ──►  4. Re-derive the numbers,
    (James, 1 min)         rebuild runs             rebuild overnight       then build the analyzers
                                                         │
                                                         └──►  process pool  ·  code review
```

**The next action is James's: run the speed probe at work** (A1 below).

| Workstream | State | Waiting on |
|---|---|---|
| **A. Processing speed** | 2 fixes shipped, 2× faster on the share | the probe at work |
| **B. More useful information** — the original goal | data capture built and shipped | the rebuild |
| **C. Review and refactor** | not started, shape agreed | the rebuild |
| **D. Checks at the shop** | 2 open | James, at the station |

---

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
      `.venv\Scripts\python scripts\ingest_speed_probe.py "\\192.168.66.9\BTXData\Departments\System Data\TEST_DATA\DLTS" 25`
      Read the `stat` figure on the first line:
      **~1 ms** → the VPN was the limit; rebuild at work, expect 5–6 hours.
      **~113 ms** → the file server (or SentinelOne's handling of the share) is
      slow even on the LAN; raise it with IT and go to A2.
- [ ] **A2 · Local mirror of the share** — now UNLIKELY to be needed (the ping
      cleared the server); only if A1 still shows slow lookups on the LAN.
      One `robocopy /MT:32 /Z` copy, restartable, then point the app at the
      copy and rebuild at local speed. *Claude writes the script.*
- [ ] **A3 · Processes instead of threads.** *After the rebuild.* Measured on 96
      real files: threads give **1.0×** at 1, 4, 8 or 16; processes give 3.6× at
      4 and **6.8× at 8**. The thread pool and its cap of 4 were built for the
      old 8 GB PC; the new laptop has 48 GB. Needs a design, not a patch:
      the processor writes "skipped" records to the database from inside the
      worker (six places); under tests each worker process would open the REAL
      database; and Windows starts processes differently from the Mac. No gain
      over the VPN (bandwidth-bound) — it pays on local files or a fast LAN.
- [ ] **A4 · Batch the saves** — after A3. Saving is one file at a time
      (~24 ms on the laptop), so it becomes the ceiling at ~40 files/sec.
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
- [ ] **B2 · Fresh-database rebuild, overnight.** *James.* Checklist: the
      2026-09-18 section of `BRING_TO_WORK.md`. Needs **12 GB free**. Blocked on
      A1. It also re-grades every final-test record — **98.5% of them (148,776
      of 151,111) are still on the old grading** — so do NOT start the separate
      15-hour re-grade; the rebuild replaces it.
- [x] **B2a · Working from home without the rebuild** (2026-09-20).
      `scripts/pull_model_slice.ps1` copies a few models' folders down the VPN
      (default 8232-1 and 8340-1, ~3 years); `scripts/build_dev_db.py` turns any
      folder into a throwaway development database with the new tables filled
      (refuses the real database by name). A 1,294-file sample database already
      exists on the Mac: `Work Files/dev_db/sample.db`, built in 69 s, 0 errors,
      839 captured trim passes. **8232-1 is the model to go deep on: 86% of its
      tracks need a second pass.**
- [ ] **B3 · Re-derive every number that rests on a final-test verdict.** After
      B2. The resistance-target figures (8232-1 correlation −0.63, yield
      10% → 73%) were computed on the old grading. **Do not set ink targets off
      them until this is done.**
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
- [ ] **B6 · The remaining eight findings**, one at a time, each with a test that it
      says nothing when there is nothing to say. The frame is built and proven; each is now a
      day's work. Next two, in order: **#2 station setup mismatch** (the laser and final test
      grading to different limit tables — D1 below is its first case, and it needs the rebuilt
      final-test data) and **#5 rework load** (hand-trim volume per model).
- [ ] **B7 · Cut-length model — PROMOTED** (James, 2026-09-20: "i want to do the
      cut length model i feel that is important"). No longer waits for the full
      rebuild: the home slice supplies real data now. Still gets its own design
      and approval, and still starts with the B4 questions — they are the cheap
      first step of this work, not a detour. Data for it:
      **laser 2 (DLTS) is the only machine recording the cut at every position.**
      8232-1's laser 2 history is 2013 → 2022-07 (4,178 files; it has run on
      laser 1 (LTS) since 2023-03). Models on laser 2 THIS year with heavy
      second-pass burden — where a recommendation could be acted on at the
      machine today: **7844 (51% need 2+ passes), 7845 (48%), 8762 (30%),
      6828 (27%)**; all of laser 2 is 28% over 11,199 tracks.

Traps the analyzers must respect (all written into the spec's known limits):
on laser 1 (LTS) the last captured pass duplicates the one before, so **pass
counts run one high** · `points_ignored_start/end` are always blank on laser 2 (DLTS)
and must never be read as 0 · `initial_trim_value` lives inside the recipe
block, not its own column.

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
- [ ] **C2… · Refactors**, from the top of that list.

## D. Checks at the shop — James

- [ ] **D1 · 8232-1: the laser and final test grade to different limit
      tables.** The only customer-facing mismatch of 36 found; James believes
      one station is set wrong. Same-day fix at the laser if so.
- [ ] **D2 · Model 8706: 41 files skipped, only 2 parsed** — the reverse of
      every other model. Open one and look for `SEC1 TRK…` sheets. If they are
      there, the parser is wrongly rejecting the model; tell Claude.
- [ ] **D3 · After the rebuild, expect the QA sweep's two red lines to
      change.** They pick "the newest 4,000 rows" by id, and a fresh database
      renumbers. Means nothing; see `BRING_TO_WORK.md`.

## E. Settings: one backlog upload instead of two inputs

James, 2026-09-20: Settings has separate "Active models" and "Pricing" inputs; "we should
simplify this and allow me to upload a current backlog to get active models and pricing
all in one." A backlog export is in `Work Files/` (gitignored — it holds customer names,
PO numbers and prices, and must never be committed).

- [ ] **E1 · Design, then build.** What a first look at the file showed: one sheet, one row
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

## Housekeeping

- [ ] **H1 · `CLAUDE.md` step 1 breaks the git remote.** There is no `.env`, so
      the command sets the remote to `https://@github.com/…` and the next push
      fails. The plain URL plus the keychain works. *James's call to delete it.*
- [ ] **H2 · Leftovers from July, never committed:**
      `docs/chart_rep_review_2026-07-16/` and `scripts/chart_rep_review.py`.
      Keep, commit or delete?
- [ ] **H3 · Read, then delete, the decision log** from the capture build —
      30 recorded judgement calls:
      `.superpowers/sdd/2026-09-17-trim-capture/progress.md`.
- [ ] **H4 · Retire V5?** It holds nothing V6 lacks; it stays until James says.

## Parked — real, not scheduled

- 35 models have a position column that does not run in order in stored
  data — a suspected ingest fault, its own investigation.
- The model specifications table is empty in production; resistance limits
  come from the laser files and the ATP documents.
- ATP audit: 36 real differences, 29 of them trim-only
  (`scripts/atp_spec_audit.py`).

## Done recently

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

- 2026-09-20 · the app now SHOWS the shop's laser names — "Laser 1 (LTS)",
  "Laser 2 (DLTS)", "Laser 3 (LTS3)" — on the company trend chart (legend in
  shop order), the Excel and unit-chart exports, and the V5 classic screens.
  Stored letters and parsing logic untouched; one function, `laser_label()`.

- 2026-09-20 · speed: one read and one lookup per file; skip warning no longer
  blames laser 3; LTS3 confirmed long since validated (547 files, 0 errors)
- 2026-09-18 · capture build shipped; disk estimate corrected 200 MB → 1.6 GB
- 2026-09-17 · repeat files stop repeating; ATP audit corrected 248 → 36
