# Process recommendations — design

Date: 2026-09-17
Status: approved in chat, not yet implemented
Owner: James Beresford

## Purpose

The app becomes an analysis tool that tells James **what to change about the
process to raise yield**. It is not a screening tool and must never be built
as one: it does not judge individual parts in flight, and no output is
intended to route or reject a unit.

That distinction changes what the same facts mean. "Incoming measurement
predicts the final verdict on 6607" is not "reject these parts". It is "the
lever for 6607 is upstream in deposition". Where incoming measurement
predicts nothing, the losses are being made at the laser instead. Every
analysis in this design is written to point at a lever, not at a part.

## The levers, and their lead times

Recommendations are worthless if they point at something nobody can move.
The engine may only speak about these, and must always state which one and
how long it takes:

| Lever | Who/where | Lead time |
|---|---|---|
| Laser settings — cut length, power, speed, passes, tolerances | At the machine, while running | Same day |
| Laser linearity limit table | At the machine | Same day |
| Ink formulation, which sets incoming resistance | Next lot of elements | Next lot |
| Deposition / upstream adjustments | Next lot; permanent change needs an ECN | Next lot, or ECN |
| ATP linearity and resistance specs | Customer drawing and specification | **Cannot change** |

The ATP row is a hard constraint, not a slow lever. The engine must never
recommend changing it.

## Finding catalogue

Each is one analyzer producing zero or more findings. The first seven need
no new parsing and could be built on today's data; the last three are what
the capture work unlocks.

**Available from stored data**

1. **Ink target.** Direction, recommended window, expected gain, how
   strongly monthly yield tracks monthly median incoming resistance, and the
   history of every past move. Lever: ink, next lot.
2. **Station setup mismatch.** Models where the laser and final test grade
   to different limit tables, or grade different amounts of travel. Per
   James, the two stations should not differ, so any difference is a setup
   error. Lever: laser limit table, same day.
3. **Where the loss is made.** Whether incoming measurement predicts the
   final verdict. Strong ⇒ the lever is upstream. Weak ⇒ the losses are made
   at the laser. Routes the reader to the right lever; never used to screen.
4. **Trim effort.** Share of each cut spent only reaching the resistance
   floor, which identifies models where the ink target is starving the
   linearity correction.
5. **Rework load.** Hand-trim volume per model, measured as trim-fail →
   final-test-pass pairs, validated against the same-unit error ratio with a
   pass/pass control group.
6. **Multi-pass burden.** Models needing second and third passes
   disproportionately. Lever: laser settings.
7. **Limit-table drift.** More than one limit table in service for the same
   model at the same station.

**Unlocked by the capture work**

8. **Configured versus optimal target.** Read the station's own incoming
   resistance limits and compare against what the data says. System A only;
   System B and C carry no configured target.
9. **Setting-change detection.** Someone changed laser power, cut length or
   indexing; did yield follow?
10. **Machine comparison.** Same model, different machine (A, B or C),
    different yield. Settings that work on one laser may not transfer.

11. **Cut-length model.** Deferred, not dropped. See its own section below.

## 1. What gets captured

**Per trim pass.** New table, one row per pass per track: the full sweep at
that stage (positions, errors, limits), measured resistance at that point,
and the recipe used for that pass (cut length, laser speeds, trim voltage,
upper and lower tolerances, ignored-point counts), plus the pass label from
the sheet. System A reads the numbered track sheets; systems B and C read
the `Trim N` / `TrimVolts N` sheets. The untrimmed sweep already lives on
the track row, so this table starts at the first cut.

Sizing: about 1.3 passes per track, roughly 114,000 rows over the full
history, adding on the order of 200 MB to a 3.7 GB database.

**Per file setup.** New table, one row per analysis, holding the whole
parameter block as stored text plus these promoted, indexed columns:
initial resistance limits, final resistance limits, laser power, pulse
duration, cut length, test voltage, indexing method, ignored-point counts.
Unpromoted fields stay queryable in the stored block.

**Deliberately not captured.** The System A `Stats` sheet is an empty
template. `Report Data` repeats the sweep the app already reads. Both cost
parse time and buy nothing.

**Format asymmetry.** System A carries explicit incoming resistance limits.
Systems B and C do not. On those models the engine computes a recommended
target instead of comparing against a configured one, and the page must say
which it is doing.

**LTS3 is already handled.** System C is format-identical to System B and is
identified by the `LTS3` folder segment in the path, not by sheet structure.
It needs no new parse path. Only the machine identity matters, and that is
already recorded.

## 2. How it is captured

- **Parser** gains two additive jobs: enumerate the intermediate pass sheets
  in order, and read the parameter block. Nothing it returns today changes.
- **Data models** gain optional fields so the processor carries the new data
  through without inspecting it.
- **Processor** change is limited to passing those fields along. No change
  to the trim path, the incremental scan, or the index load.
- **Manager** owns the new tables and their writes, created by the existing
  idempotent migration pattern.
- **Final-test save early return is fixed.** Today a re-read file whose
  content matches a stored row returns early and updates nothing, so a
  reprocess silently refreshes trim rows but not final-test rows. It must
  update in place, so reprocesses are repeatable.

## 3. Migration and the run at work

Tables arrive via migration on startup; no rebuild.

**The reprocess goes into a fresh database.** A reprocess over the existing
one would capture only part of what is wanted, because of the early-return
behaviour above. A fresh database also:

- captures per-pass sweeps and parameter blocks for every file;
- regrades every final-test record with the ignore-window fix, which makes
  the pending re-grade unnecessary (roughly fifteen hours saved);
- clears the legacy verdict count to zero rather than chipping at 149,000
  rows.

Nothing is lost. Everything in the database is derived from the files.
Trained models live in their own folder and survive.

**Sequence:** pull; back up the current database and move it aside; launch so
the new schema is created empty; run every folder with incremental
unchecked; retrain when it finishes. Expect four to eight hours — an
overnight job.

## 4. The engine and its surfaces

**A finding** carries: model, machine or machines, category, lever, lead
time, expected gain, strength of support, units it rests on, and an evidence
bundle. Everything downstream reads that one shape, so a new analysis means
a new analyzer, not a change to the page.

**Expected gain** is defined per analyzer and must be stated in the
finding, not left implicit. For the ink target it is the combined yield
inside the recommended window minus the model's current overall combined
yield, where combined means passed linearity and landed inside the final
resistance limits. Analyzers that cannot express a gain honestly report
none, and such findings rank below any that can.

**Strength of support** is likewise analyzer-defined and named in the
finding — a correlation coefficient for the ink target, a separation
measure for "where the loss is made" — never a bare adjective.

**Ranking** is expected gain in yield points times annual volume, so the top
of the list is units per year recoverable, not percentages. Findings with no
expressible gain sort last, ordered by volume.

**Two surfaces, one engine.** A ranked cross-model list is the front door.
Investigate shows the same findings filtered to the model in view.

**Evidence tiers.** Every finding carries its curve, its sample size and its
strength. Findings on models above a configurable volume threshold —
defaulting to 500 trimmed tracks in 24 months — also carry the full pack:
every past time that lever moved and what happened, and what was ruled out.

**It must be able to say nothing.** Weak relationship, thin sample,
mismatched axes, or "this lever is not your problem here" are all stated
plainly. A model with no findings reads as nothing to do, not as a gap. No
analyzer may manufacture a recommendation to fill space.

**When it runs.** Findings are computed after an ingest, alongside the
existing drift advance, and cached. Opening the app is instant.

**Out of scope for this build:** dismissing or snoozing findings. Likely
wanted once the list has been lived with; not before.

## 5. Testing

- **No-op proof for the parser.** Several hundred real files through the old
  tree and the new one into separate databases; every stored field diffed;
  zero differences, with the new tables populated alongside.
- **Every new check fails first**, demonstrated against current code before
  it enters the sweep.
- **Negative tests per analyzer.** Each is fed a model where the
  relationship is genuinely absent and must produce nothing. Silence is a
  tested behaviour.
- **Arithmetic pinned to hand-computed cases** worked out on 2026-09-17: the
  ink correlation, the trim-effort share, the rework signature and its
  control group.
- **Falsification mode.** The engine's self-check breaks its own inputs and
  confirms the checks go red.
- **Fixtures.** Real sample files for both formats are already in the repo.
  LTS3 needs no parse fixture; only the folder-based machine detection needs
  a test.
- **Database safety unchanged.** Read-only against the work database;
  harnesses run on copies and refuse the real path by name; copies deleted.

## Build order, and how it is planned

1. **Capture.** Parser, data models, processor pass-through, manager tables,
   early-return fix. Ends with the no-op proof green.
2. **Reprocess at work** into a fresh database.
3. **Engine frame plus the ink-target analyzer** end to end, both surfaces.
4. **Remaining analyzers**, one at a time, each with its negative test.

This is too much for one implementation plan, and the middle step is a
several-hour operation on James's machine that everything after it depends
on. So it is planned in two halves: **the first implementation plan covers
step 1 only**, ending at the no-op proof. Steps 3 and 4 get their own plan,
written after the reprocess has run and the trajectory data can actually be
looked at — which may well change what the later analyzers should do.

## The cut-length model — deferred, not dropped

This is the most valuable thing the per-pass data could give us, and James
has asked explicitly that it not be lost. It is deferred only because it
cannot be designed before the data it learns from exists.

**What it is.** The per-pass table gives, for every cut ever made: the curve
going in, the cut applied (length, speeds, trim voltage, tolerances), and
the curve coming out. That is a supervised learning problem with the
label already recorded. Learn the mapping and you can recommend the cut
rather than score the outcome after the fact.

**Why it has to wait.** The trajectory data does not exist in the database
yet. Designing the model from the single file opened on 2026-09-17 would be
guessing. Once step 2 has run, roughly 114,000 pass rows exist across the
full history, of which the multi-pass tracks — about 5,300 in the last two
years — carry the before-and-after pairs that matter most.

**The first questions to ask of it, before any modelling.** These are cheap
and may make a model unnecessary:

1. Does the first pass systematically make linearity worse? In the one file
   examined, worst error went 0.163 before trimming, 0.191 after pass one,
   then 0.063 after pass two. If that pattern is general it is a finding on
   its own and a same-day laser lever.
2. For a given starting shape, which cut lengths produce a pass on the first
   attempt and which lead to a second?
3. Is there a cut that would have avoided the second pass entirely? Second
   and third passes are 23% of tracks and pure cost.
4. Do the answers differ between the three machines?

**Entry criteria.** Pick this up when the per-pass table is populated from a
completed reprocess, and after questions 1 to 4 have been answered
descriptively. It gets its own design pass and its own approval; it is not
carried by this spec's approval.

## Known limits, carried forward

- Role of a table in an ATP (trim versus final test) is unstated on most
  data sheets. Anything resting on that is conditional.
- The model specifications table is empty in production; resistance limits
  come from the laser files, and from the ATP prose where a file lacks them.
- 35 models have a non-monotonic position column in stored data — a
  suspected ingest fault, its own investigation, out of scope here.
- Trim-side units are unconfirmed against ATP tables. Range and shape
  comparisons are unit-independent and safe; absolute value comparisons on
  the trim station are not.
- **A reprocess refreshes a final-test row's verdict and tracks, but not its
  trim link, match confidence, or resistance fields** (added 2026-09-17,
  Task 8). Those are set only on first insert. Moot for the fresh-database
  route in section 3, where every column is written once. On a later
  reprocess over an already-populated database a refreshed verdict can sit
  beside a trim link from the previous parse, so any analyzer that reads
  those columns together must not assume they came from the same parse.
