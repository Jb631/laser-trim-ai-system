# Tracker

What is open, in what order, and who holds the next move. Updated in the same
commit as the work it describes. `BRING_TO_WORK.md` stays the place for
step-by-step instructions at the work machine; this is the index above it.

Last updated: 2026-09-20

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
- [ ] **B5 · Recommendation engine + the ink-target finding**, end to end, on
      both screens. Gets its own plan, written after B2 so it is designed
      against real data.
- [ ] **B6 · The other nine findings**, one at a time, each with a test that it
      says nothing when there is nothing to say.
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
on lasers 1 and 3 (LTS, LTS3) the last captured pass duplicates the one before, so **pass
counts run one high** · `points_ignored_start/end` are always blank on laser 2 (DLTS)
and must never be read as 0 · `initial_trim_value` lives inside the recipe
block, not its own column.

## C. Review and refactor

Agreed shape: **not** one big rewrite — the outputs go to customers. A review
that produces a ranked list by what each problem costs, then the worst items
one at a time, each proven against the 645-file baseline. *Starts after B2.*

- [ ] **C1 · The review itself** → a ranked list. Evidence already in hand:
  - the same hash bug existed three times, once per parser (fixed `0add30e`)
  - `database/manager.py` is over 10,000 lines
  - ~70 call sites use a global database handle that ignores a test's
    database and opens the real one
  - `generate_plots` is a parameter nothing reads
  - several QA-sweep checks could not fail (fixed), and
    `scripts/app_qa_sweep.py` is over 3,000 lines
  - trained models were saved under scikit-learn 1.8.0 and load under 1.9.0
    on the Mac with a version warning
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

- 2026-09-20 · the app now SHOWS the shop's laser names — "Laser 1 (LTS)",
  "Laser 2 (DLTS)", "Laser 3 (LTS3)" — on the company trend chart (legend in
  shop order), the Excel and unit-chart exports, and the V5 classic screens.
  Stored letters and parsing logic untouched; one function, `laser_label()`.

- 2026-09-20 · speed: one read and one lookup per file; skip warning no longer
  blames laser 3; LTS3 confirmed long since validated (547 files, 0 errors)
- 2026-09-18 · capture build shipped; disk estimate corrected 200 MB → 1.6 GB
- 2026-09-17 · repeat files stop repeating; ATP audit corrected 248 → 36
