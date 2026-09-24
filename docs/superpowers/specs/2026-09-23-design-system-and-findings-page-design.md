# V6 facelift, step 1: the new look app-wide, and the Findings page rebuilt

**Status:** approved section by section by James, 2026-09-23. This document is the
authority for the build plan; where the plan and this disagree, this wins.

**What James asked for (2026-09-23):** "i think the app needs a facelift or polish, i dont
like the design, lastly the finings sheet i dont like the layout its just a bunch of rows
and its hard to see whats important."

## Decisions, in the order James made them

| # | Question | James chose |
|---|---|---|
| 1 | What should the Findings page answer first? | "biggest impact? or grouped by issue?" → shown both → **grouped by issue (layout B)** |
| 2 | What is wrong with the current look? | **dated or generic**, and **hard to read** (not: too dark, not: cramped) |
| 3 | Which visual direction? | **3 · refined dark** — today's navy, with brighter text, bigger numbers, one accent |
| 4 | How far should the facelift go? | **restyle AND relayout every page** — delivered in steps, this is step 1 |
| 5–8 | Sections 1–4 below | approved as written |

## The problem, measured

**The Findings page buries what matters.** 96 findings across 62 models, every one drawn
as an identical full-width button. Only **9 claim a yield gain** (7 cut-setting, 2 ink
target). The other 87 are context — 48 trim-effort facts on their own are half the page —
and all 96 look the same, so the ~300-tracks-a-year 6607 recommendation is visually
identical to the smallest fact on the list.

**Three of today's colour pairs fail the basic readability minimum** (WCAG contrast 4.5:1):

| pair | today | new |
|---|---|---|
| body text on a card | 11.1 | 13.4 |
| secondary text on a card (most explanatory writing) | 5.4 | 8.0 |
| secondary text on the page | 6.8 | 9.7 |
| **dimmed text on the page** | **2.8** | 5.0 |
| **accent colour on a card** | **3.5** | 8.0 |
| **out-of-control drift tier on its own background** | **4.2** | 6.2 |

**The app has no colour for PASS or FAIL.** The theme holds only drift-watch tiers; the
linearity verdict — the most important thing the app reports — is plain text everywhere.

**Every page title is an 11 px all-caps label in the accent colour**
(`WHAT TO CHANGE, BIGGEST FIRST`): the hardest text to read on the screen, and much of the
"dated" feel.

## How the whole facelift is divided

James chose to restyle and relayout every page. That is too much for one design, and every
relayout depends on a shared look being settled first, so it is delivered in steps, each
designed and approved before it is built:

1. **This document:** the new look everywhere (tokens, fonts, building blocks), plus the
   Findings page — the first page rebuilt on it.
2. **Then each remaining page**, one at a time, each with its own short "what matters on
   this page" round, reusing step 1: Model (1,259 lines, the densest), Home, Dashboard,
   Triage, Process, Settings. Order to be agreed with James after step 1 ships.

A separate project, not part of the facelift: capturing laser 1's `TrimVolts` sheets
(TRACKER B1a). It gets its own design.

---

## Section 1 — Tokens

Every colour, font and size in V6 already comes from one place, `gui/v6/theme.py`
(`ThemeManager`). The new look is a change of VALUES under the existing names, so every
page picks it up at once. Measured before this was written: only **3** hard-coded hex
colours exist outside `theme.py` (all in `widgets/company_trend_chart.py`), and the charts
already read their colours from the theme.

### Colours — refined dark

| token | today | new |
|---|---|---|
| `BG` | `#1a1f2e` | `#111a28` |
| `SURFACE` | `#1e2435` | `#172233` |
| `CARD` | `#263244` | `#1c2a3e` |
| `ELEVATED` | `#2f3b50` | `#243550` |
| `SIDEBAR_BG` / `SIDEBAR_ACTIVE` | `#1a1f2e` / `#263244` | `#111a28` / `#1c2a3e` |
| `SIDEBAR_STRIPE` | `#3b82f6` | `#4fd6b8` |
| `ACCENT` | `#3b82f6` | `#4fd6b8` |
| `ACCENT_HOVER` / `ACCENT_PRESSED` | `#60a5fa` / `#2563eb` | `#74e0c8` / `#36b99c` |
| `TEXT_PRIMARY` | `#e8eef5` | `#f3f6fa` |
| `TEXT_SECONDARY` | `#9ca8bd` | `#b6c2d2` |
| `TEXT_DISABLED` | `#5a6478` | `#7b8aa0` |
| `TEXT_INVERSE` | `#1a1f2e` | `#0b1f1b` (dark text for teal fills) |
| `DIVIDER` / `BORDER` | `#2a3142` / `#3a4456` | `#26344b` / `#34465f` |

**New tokens:**

| token | value | use |
|---|---|---|
| `ACCENT_TINT` | `#123a37` | background of teal count pills |
| `CHECK` / `CHECK_TINT` | `#ff8f7a` / `#3e2522` | "check this" — coral pill, error banner |
| `PASS_FG` / `PASS_BG` | `#9bd66f` / `#1f3322` | linearity PASS badge (7.9:1) |
| `FAIL_FG` / `FAIL_BG` | `#ff8f7a` / `#3e2522` | linearity FAIL badge (6.3:1) |
| `NEUTRAL_FG` / `NEUTRAL_BG` | `#c3cedb` / `#243550` | UNTRIMMED / NOT GRADED badge, evidence tags (7.7:1) |
| `WATCH_FG` / `WATCH_BG` | `#f5b544` / `#3a2f16` | SIGMA WATCH badge (7.2:1) |
| `CHART_REFERENCE` | `#8a9bb3` | the grey "previous / reference" chart series (5.1:1 on a card) |

**The drift tiers keep their meanings.** Measured on their own backgrounds: `TIER_WARNING`
6.0:1 and `TIER_DRIFT` 5.1:1 pass and keep their values. **`TIER_OOC` fails today** — the
most severe drift signal in the app, at 4.2:1 on `TIER_OOC_BG` and 3.8:1 bare on a card — so
it changes `#ef4444` → **`#ff7a7a`** (6.2:1 on `#3d1818`, 5.7:1 on a card). It sits close to
the FAIL coral in hue; that is acceptable because out-of-control appears on drift views and
FAIL on verdict badges, and every badge carries its word.

**`TEXT_INVERSE` is safe to change:** all 18 uses in `gui/v6` sit on `ACCENT`-filled buttons
(checked 2026-09-23), where the new dark value measures 9.5:1 on the new teal.

**Why these hues.** The PASS green sits at 94° of hue and the teal accent at 167° — 72°
apart — so "this passed" and "act here" never read as the same signal. Sigma watch is
amber, not red, on purpose: CLAUDE.md's domain rule is that sigma is a drift-watch signal
and never a rejection, so it must not look like a FAIL.

### Type — IBM Plex, bundled

- **IBM Plex Sans** for words; **IBM Plex Mono for every number** — model numbers, counts,
  rates, settings, readouts — so figures line up in columns and a 6 is never mistaken for
  an 8.
- **Bundled with the app** (regular and medium weights of each, with the SIL Open Font
  Licence 1.1 file beside them) under `src/laser_trim_analyzer/gui/v6/fonts/`, loaded at
  startup through `customtkinter.FontManager.load_font`. Verified in the installed
  CustomTkinter 5.2.2: on Windows it loads the file privately (`AddFontResourceEx`,
  private, not enumerable) — no install, no admin rights, no IT request. On macOS it
  returns False, so the Mac development copy falls back.
- **Fallback, never silent failure:** if loading fails, the family falls back to
  `("Segoe UI", "system-ui")` for Sans and `("Cascadia Mono", "Consolas")` for Mono, and one
  WARNING line names the file that failed.
- `FONT_FAMILY` becomes `("IBM Plex Sans", "Segoe UI", "system-ui")`; a new `MONO_FAMILY`
  is `("IBM Plex Mono", "Cascadia Mono", "Consolas")`.

**Scale, one step up everywhere:**

| token | today | new |
|---|---|---|
| `SIZE_CAPTION` | 11 | 12 |
| `SIZE_BODY` | 13 | 14 |
| `SIZE_HEADING` | 16 | 17 |
| `SIZE_TITLE` | 20 | 22 |
| `SIZE_DISPLAY` | 28 | 30 |
| `SIZE_READOUT` *(new, mono)* | — | 20 |

The **32 hard-coded font sizes** outside `theme.py` move onto this scale, so nothing stays
small when everything else grows.

### Spacing and radii

Unchanged — `SPACE_XS…2XL` (4–32) and `RADIUS_SM/MD/LG` (4/6/8) are already a sensible
4-px grid.

### Rules that come with the tokens

1. **Teal fills carry dark text** (`TEXT_INVERSE`). White on this teal measures 1.8:1;
   dark measures 9.5:1.
2. **Colour means something or it is not used.** Teal = act here. Coral = check this.
   Green / coral / grey / amber = the four verdicts. Nothing is coloured for decoration.
3. **Numbers are always mono.**

---

## Section 2 — Building blocks

Shared pieces, built once in `gui/v6/widgets/` and reused by every page relayout in the
later steps. Each is drawn from the Section 1 tokens only.

| block | what it is |
|---|---|
| **Page header** | Page title, sentence case, `SIZE_TITLE`, `TEXT_PRIMARY`; one caption line under it in `SIZE_BODY`, `TEXT_SECONDARY`. Replaces `PageBase._zone_header`'s 11 px all-caps accent label on every page. |
| **Group header** | Label (`SIZE_HEADING`), a count pill after it, and on the right the unit of that group's numbers; one line under it saying what the group means. A hairline under the header. |
| **Row** | Three columns: model number (mono, `TEXT_SECONDARY`) · statement (`SIZE_BODY`) · readout (right-aligned). The whole row is clickable and lifts to `ELEVATED` on hover. Rows are separated by hairlines, not each boxed in a card — boxing everything identically is much of why nothing stands out today. |
| **Readout** | A mono number at `SIZE_READOUT`, weight medium, `TEXT_PRIMARY`, with an optional small unit in `SIZE_CAPTION`. Wherever the number is the point. |
| **Count pill** | Mono count on a tint: `ACCENT`/`ACCENT_TINT` for things to act on, `CHECK`/`CHECK_TINT` for things to check. |
| **Verdict badge** | **PASS**, **FAIL**, **UNTRIMMED** / **NOT GRADED**, **SIGMA WATCH** — always the word AND the colour, never colour alone, so it works for colour-blind readers and PASS can never be confused with the accent. |
| **Tag** | Small `NEUTRAL_FG`/`NEUTRAL_BG` label beside a statement — "both tracks", "two periods · test first". |
| **Buttons** | At most ONE teal-filled button per screen, for the main action, with `TEXT_INVERSE` text. Everything else outlined or plain text. |
| **Tabs** | Sentence case; the active tab is marked by a teal underline, not a filled block. |
| **Tables** | Numbers mono and right-aligned; hairline rows. |
| **Charts** | Teal (`ACCENT`) for the main series, `CHART_REFERENCE` grey for previous or reference, coral (`CHECK`) for limit lines and failing points, `DIVIDER` for gridlines (deliberately faint, 1.2:1), `TEXT_SECONDARY` for axis text. Measured on a card: teal 8.0, grey 5.1, coral 6.5. |
| **Empty and error states** | The project rule stands — a failure must never look like a result. Errors: a `CHECK_TINT` banner naming what failed. Empty: one line saying what the space is for and what would fill it. |

---

## Section 3 — The Findings page

Visual reference: `docs/superpowers/specs/2026-09-23-findings-page-mockup.html` (a web
rendering of the approved design with real data from the 2026-09-22 rebuild). The CTk build
matches its colours, type and hierarchy; shapes follow it as closely as CustomTkinter
allows.

### Page header

Title **Findings**. Caption, built from the row counts after merging:

> `{a} changes worth testing · {b} ways to save laser time · {c} tests to check · worked out {date}`

`{date}` is the newest `computed_at` across the findings, so James can see how fresh they are.

### Four groups, in this order

Every analyzer declares which group its findings belong to. **An analyzer with no declared
group must fail a test, never vanish from the page** — the same rule as everywhere else: a
failure must not look like a result.

| group | analyzers (category) | readout column | sort |
|---|---|---|---|
| **Change a setting to raise yield** — *a different setting did better on the same test* | `cut_setting` (Cut setting), `ink_target` (Ink target) | **tracks a year** (`tracks_per_year`); a row with no rate shows "—" | by tracks a year, then rows without a rate by `n_units` |
| **Laser time you could save** — *units cut that didn't need it, or given more cuts than planned* | `trim_effort` (Trim avoidance, Pass effectiveness), `pass_burden` (Multi-pass burden) | **tracks**: Trim avoidance → `arrive_in_spec_n`; Pass effectiveness → `multi_cut_n`; Multi-pass burden → `tracks_over_recipe` | by that count |
| **Check the test** *(coral pill)* — *graded against more than one limit table, so pass rates across the change don't compare* | `limit_tables` (Limit table) | **tracks** (`n_units`) | by tracks |
| **What changed** — *recipe changes, newest first* | `recipe_change` (Setting change) | **pass-rate move** in points (after − before), green if up, coral if down | by date of change, **newest first** — it is a history |

Multi-pass burden's readout is `tracks_over_recipe`, not `unplanned_passes`, so every
number in the "laser time" column is a count of tracks; the pass count goes in the opened
row.

### Rows

- **Statement** names the laser in shop terms ("Laser 1: cut 6900 → try 6800") — never the
  code's letters.
- **Merging:** findings merge into ONE row when they share the model, the analyzer and the
  recommendation, differing only by track name. The readout is their sum (tracks are
  additive), a "both tracks" tag says so, and the opened row lists each track's own numbers.
  6607's two cut-setting findings (~182 and ~118 a year) become one row, ~300.
- **Evidence tag:** an analyzer may supply one; the page shows it beside the statement.
  `cut_setting` supplies its grade: *same days* (settings mixed on ≥25% of production
  days), *side by side* (both in use in at least half the months either was), or *two
  periods · test first*.
- **Five rows per group** (`ROWS_PER_GROUP = 5`), then **Show all {n}**, which expands the
  group in place.

### Opening a row

**Clicking a row opens it in place** — it no longer jumps to the model page. The evidence
and the test to run are what make a finding usable, and today they are only visible after
leaving the list. One row is open at a time; clicking it again, or another row, closes it.

An opened row shows, from what the finding already stores:
1. the finding's own `summary`;
2. for `cut_setting`, the settings table from `evidence.group.settings` — setting, tracks,
   in spec — one column per track when rows were merged;
3. **Open {model}** — the page's one teal button — which goes to that model's Findings tab.

### States

All the current honesty is kept: a failed load shows a `CHECK_TINT` banner with the reason
("this is an error, not an empty list"); models that could not be worked out on the last
refresh are named; the "could not check which models failed" case is still reported. **All
four groups are always shown on the page**; an empty one says what would put something in
it.

### The model page's Findings tab

`widgets/findings_tab.py` uses the same group headers and rows, filtered to one model, so
the same data never looks two different ways. On a single model's tab, empty groups are
hidden; if all four are empty it keeps today's line ("nothing to act on — that is a result,
not a gap"). The tab's "what was measured" facts are restyled with the Section 2 blocks; the
rest of the model page keeps its layout until its own step.

### The two analyzer fixes that ship with this

Found on the rebuilt database (2026-09-22) and promised to James:

1. **A setting must have run for a real period before it can be "best".** 8397-2's "best"
   setting, 1.7107, ran for 23 days and 80 tracks; the model's sustained settings say the
   opposite (1.3 → 82%, 1.4 → 68%, 1.5 → 51%). `cut_setting` gains a minimum span of
   production days for a setting to be eligible as the winner — proposed 60 days, to be set
   from the data in the plan. A short run still appears in the opened row's settings table
   as context.
2. **"Now running" must mean now.** 8397-2 has no files since April 2025, yet its finding
   says "the 1.5 now running". When a group's newest track is more than 180 days older than
   the newest trim file in the database, the finding says "last ran at 1.5 (Apr 2025)" and
   claims no rate.

---

## Section 4 — Rollout and verification

### Order of work (each step tested before the next)

1. Bundle IBM Plex with its licence; load at startup with the logged fallback.
2. Swap the Section 1 token values; add the new tokens.
3. Move the 32 hard-coded font sizes onto the scale; fix the 3 hard-coded chart colours.
4. Build the Section 2 blocks as shared widgets.
5. Rebuild the Findings page and the model page's Findings tab on them.
6. The two analyzer fixes.

### What every other page gets in this step

The new look only: palette, fonts, bigger text, and sentence-case titles through the new
page header. Their layouts are untouched. **The main risk** is that larger text in
unchanged layouts truncates or wraps — most likely on the Model page. Overflows found in
this step are fixed in this step (with the smallest change that fixes them); rearranging a
page waits for that page's own step.

### How we know it is done

- **The whole test suite** (`scripts/run_test_gate.py`), green.
- **The chart render sweep** (`scripts/chart_qa_render_all.py`) on a copy of the database —
  and every rendered PNG is looked at, not just the exit code.
- **The app QA sweep** (`scripts/app_qa_sweep.py`) on a copy; new features get a sweep
  entry in the same commit.
- **New — a readability test.** Asserts every text-on-surface pair the theme uses meets
  4.5:1, `TEXT_PRIMARY` on every surface meets 7:1, and `TEXT_INVERSE` on `ACCENT` meets
  4.5:1 — so "hard to read" cannot come back through a later colour change.
- **New — fonts test.** The bundled files and their licence exist; a failed load falls back
  and logs rather than raising.
- **New — Findings tests.** Every analyzer maps to a group (an unmapped one fails); the
  merge rule; each group's sort; Show all; one row open at a time; every existing
  error and empty state still appears. Each silence or refusal is made to fail first.
- **Every page is opened against a copy of the database and looked at** before anything is
  called finished. A green test run does not catch a label cut in half.

### At work, James checks

1. The font is really Plex — the mono numbers are the giveaway.
2. Nothing is cut off, on any page.

## Out of scope for step 1

- Relayouts of Home, Model, Dashboard, Triage, Process and Settings (steps 2+).
- The V5 classic screens — untouched, they stay as the fallback.
- A light theme.
- Automatically flagging batches of recipe changes, such as laser 1's October 2024 (6126
  and 1844205 on the 2nd, 6607 on the 4th) and July 2026 (8232-1, 1844205). Noted as a
  possible future finding.
- Recording why ERROR rows failed (none of the 237 on the rebuilt database carries a
  reason) — a processing fix, tracked separately.
- `TrimVolts` capture — its own design.
