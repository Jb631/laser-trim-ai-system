# Tracker

What is open, in what order, and who holds the next move. Updated in the same
commit as the work it describes. `BRING_TO_WORK.md` stays the place for
step-by-step instructions at the work machine; this is the index above it.

Last updated: 2026-09-20

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
- [x] **Speed probe** — `scripts/ingest_speed_probe.py`, times each layer
      separately, writes only to a throwaway database.
- [ ] **A1 · Run the probe AT WORK, on the office network.** *James · 1 minute.*
      `.venv\Scripts\python scripts\ingest_speed_probe.py "\\192.168.66.9\BTXData\Departments\System Data\TEST_DATA\DLTS" 25`
      Read the `stat` figure on the first line:
      **~1 ms** → the VPN was the limit; rebuild at work, expect 5–6 hours.
      **~113 ms** → the file server (or SentinelOne's handling of the share) is
      slow even on the LAN; raise it with IT and go to A2.
- [ ] **A2 · Local mirror of the share** — only if A1 says the server is slow.
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
      settings, and on laser 1 the per-position cut length, trim current and
      predicted-vs-actual correction. 17 commits, `adcc1ce..5f94dc4`. Proven to
      change nothing that existed: 645 real files, 26 stored columns.
- [ ] **B2 · Fresh-database rebuild, overnight.** *James.* Checklist: the
      2026-09-18 section of `BRING_TO_WORK.md`. Needs **12 GB free**. Blocked on
      A1. It also re-grades every final-test record — **98.5% of them (148,776
      of 151,111) are still on the old grading** — so do NOT start the separate
      15-hour re-grade; the rebuild replaces it.
- [ ] **B3 · Re-derive every number that rests on a final-test verdict.** After
      B2. The resistance-target figures (8232-1 correlation −0.63, yield
      10% → 73%) were computed on the old grading. **Do not set ink targets off
      them until this is done.**
- [ ] **B4 · Four cheap questions about the trim passes, before any model:**
      does the first pass make linearity worse (one file: 0.163 → 0.191 →
      0.063)? · which cut lengths pass first time? · was there a cut that
      avoids the second pass (2nd and 3rd passes are 23% of tracks)? · do the
      three lasers differ?
- [ ] **B5 · Recommendation engine + the ink-target finding**, end to end, on
      both screens. Gets its own plan, written after B2 so it is designed
      against real data.
- [ ] **B6 · The other nine findings**, one at a time, each with a test that it
      says nothing when there is nothing to say.
- [ ] **B7 · Cut-length model.** James's priority — deferred, NOT dropped. Entry:
      B2 done and B4 answered. Gets its own design and approval.

Traps the analyzers must respect (all written into the spec's known limits):
on lasers 2 and 3 the last captured pass duplicates the one before, so **pass
counts run one high** · `points_ignored_start/end` are always blank on laser 1
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

- 2026-09-20 · speed: one read and one lookup per file; skip warning no longer
  blames laser 3; LTS3 confirmed long since validated (547 files, 0 errors)
- 2026-09-18 · capture build shipped; disk estimate corrected 200 MB → 1.6 GB
- 2026-09-17 · repeat files stop repeating; ATP audit corrected 248 → 36
