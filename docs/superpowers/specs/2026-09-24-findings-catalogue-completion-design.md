# The findings catalogue, completed (TRACKER B6, B1c) — 2026-09-24

James, 2026-09-23 (`/goal`): "finish the redesign, complete all the parse upgrades and fixes &
finish the task list" — decided here as rulings (*what — why — what it costs if wrong*) for James
to overturn. The catalogue is `docs/superpowers/specs/2026-09-17-process-recommendations-design.md`
("Finding catalogue"). Built so far: #1 ink target, #4 trim effort, #6 multi-pass burden, #7 limit
tables, #9 recipe changes (the cut recipe), and the cut-setting analyzer. This document finishes
the rest. Every number was measured read-only on the rebuilt work database (2026-09-23/24).

The app recommends process changes; it never screens parts. Every analyzer below keeps the
engine's rules: it can say nothing; a crash is named in `facts["errors"]`; a finding claims a
gain only with a `gain_definition` and its own population; the only levers are the four in
`findings/model.py`; lasers are named with `laser_label()`.

## What is already half built

- **#2 station setup mismatch** — `core/spec_alignment.compare_station_specs` already answers
  "do the laser and final test grade this model to the same limits?" by the position-matched,
  knee-aware method (the naive comparison called 81 models mismatched; this one 18). It drives
  the Model page's spec banner and the Triage list. It is not a finding yet.
- **#8 configured versus optimal target** — `ink_target` already reads the station's configured
  incoming window (laser 2 (DLTS) files carry one) and quotes it in its summary. It never says
  when the configured window and the recommended one disagree.
- **#9 setting changes** — `recipe_change` covers the cut recipe. The other 40-odd laser settings
  stored per file in `trim_setup.parameters` (TRACKER B1c: "captured and analysed by nothing")
  are the rest of #9.

## 1. Station setup mismatch (#2) — `station_setup`

**Ruling 1:** a new analyzer that turns a `differs` comparison into a finding in the **Check the
test** group: lever `laser_limit_table`, no gain claimed, readout = the model's graded tracks in
its latest year. Statement: "{laser}: the laser grades to limits about N× wider/narrower than
final test over P% of the travel" (N and P from the comparison). The analyzer calls the
comparison's sampling and comparison functions itself rather than `compare_station_specs`,
because that function turns a read failure into "insufficient" for the banner's sake — here a
failure must reach `facts["errors"]`. `insufficient` and `aligned` say nothing — why: James,
2026-08-30, "i also want to know when the trim and test specs dont align", and TRACKER D1 names
8232-1 as the customer-facing case — cost if wrong: a check-the-test row per mismatched model.

## 2. Where the loss is made (#3) — a fact, and a finding only when strong

**Ruling 2:** per laser, how well the INCOMING measurement (the untrimmed sweep's largest error
magnitude, and the incoming resistance) separates tracks that end the laser inside limits from
those that do not, as an AUC on the latest year. Stored in `facts["loss_origin"]` for the Model
page's Findings tab, always. A finding (Change a setting group, lever `deposition`, no gain) only
when the AUC is ≥ 0.70 on ≥ 300 tracks with at least 50 of each outcome: "{laser}: incoming
linearity predicts the laser verdict (AUC 0.74) — the loss starts before the laser". Measured
2026-09-24 (latest year): 6607 on laser 1 (LTS) 0.74 over 1,290 tracks (465 fail) — the one
model that qualifies of eight checked; 8232-1 0.57 and 8340-1 0.57 (made at the laser, as the
2026-09-17 study found); 8202-1 0.71 but only 27 failures, which the 50-per-outcome floor
excludes. A weak AUC is a fact, not a finding: the other
analyzers already cover the laser's own levers. The Model page's predictor panel reports the ML
predictor's AUC on final test; the two sit together on the tab — why: this routes James to the
right lever and never grades a part — cost if wrong: one sentence on a tab.

## 3. Rework load (#5) — `rework_load`

Measured: 48,559 of 151,793 final-test records are linked to a trim analysis. In the last year
the laser-FAIL → final-test-PASS signature — units hand-trimmed after failing at the laser,
proven by the same-unit error ratio against a pass/pass control group (memory
`overkill-retrim-confound`) — is 580 of 1,094 linked on 8340-1, 506 of 1,299 on 6607, 424 of
1,272 on 8232-1, 418 of 594 (70%) on 8397-2, and near zero on most models.

Those figures are a naive first look (each final-test record against its one linked analysis).
The app already has THE definition of a unit's trim disposition — the unit-DAY's: per track the
last attempt of the day, then every track must pass (`get_model_trim_ft_agreement`, 2026-08-30) —
and a one-file-per-attempt model counts a unit's early failed attempts otherwise.

**Ruling 3:** a new analyzer in the **Laser time you could save** group (its meaning widens to
"…or hand-trimmed after failing at the laser"), counting unit-days through that shared
definition, never its own join: readout = unit-days in the latest year whose laser disposition
is FAIL and whose final test PASSES; lever `laser_settings`; no gain claimed. It says nothing
unless the control-group ratio confirms the signature (the rework pairs' final-test error is
materially below their own laser error, and the pass/pass pairs' is not) — why: hand trim is
labour the laser's failures create, and the company's stated goal is to "trim as little as
possible" — cost if wrong: one group's meaning is one clause longer.

## 4. Configured versus optimal target (#8) — inside `ink_target`

**Ruling 4:** no new analyzer. When a configured incoming window exists and the recommended
window lies outside it (no overlap), `ink_target`'s finding gains the tag "outside the configured
window" and one sentence saying so; the evidence records both windows — why: it is the same
comparison the analyzer already makes — cost if wrong: none.

## 5. Machine comparison (#10) — `machine_compare`

Measured: only four models ran on two lasers in the same months with ≥ 20 graded tracks each
since 2023-09 — 6126 laser 2 (DLTS) 99.1% of 335 against laser 1 (LTS) 73.7% of 453; 6952; 8821;
7458-1.

**Ruling 5:** a new analyzer in the **Change a setting** group: for each limit table the lasers
SHARE, in the months both lasers ran the model, compare the pass rates; a finding when the gap is
≥ 10 points with ≥ 100 tracks on each side, lever `laser_settings`, no gain claimed ("settings
that work on one laser may not transfer"). Never across limit tables — a pass rate is a verdict
against a test — why: the catalogue's #10 — cost if wrong: a handful of rows.

## 6. The other laser settings (#9, B1c) — `setup_change`

**Ruling 6:** a new analyzer in the **What changed** group, the `recipe_change` pattern applied
to the stored setup parameters: when a numeric parameter (laser power, speeds, frequency, trim
voltage, tolerances, indexing…) changes value for a (model, laser) and the new value runs ≥ 60
days, report the pass-rate move across the change with the limit table held constant. Keys are
read through one alias table so laser 1's `Response` and laser 2's `Response (Linear or
Function)` count as one setting (the 2026-09-23 ruling). Identity-like keys (serial, dates, file
names) are excluded by an explicit list — why: TRACKER B1c — cost if wrong: a noisy history group,
which the 60-day rule and the limit-table rule keep small.

## Presentation

Every new analyzer is mapped in `findings/presentation.py`'s `ANALYZER_GROUP` (the test that
every analyzer module has a group enforces it) and gets a statement and readout rule there.

## Verification

Each analyzer ships with: a test that it says nothing on data with nothing to say (made to fail
first), a test per rule and threshold, a test that a crash is named, the whole suite green, and
a run on a COPY of the work database whose output is recorded in the commit.
