# Option B and the finish pass — design (2026-10-04)

James, 2026-10-04, with Task Manager TMOG open: "now this is an example of a finished peice of
solftware, this is what im going for with this laser trim app" — then, shown three TMOG-style
versions on his real data: "i like B but im not saying we need to copy everything, im just saying
TMOG feels more like a finished peice of software". TMOG is the bar for FINISH, not a template.

## Decisions James made
1. **B**: keep the top bar he picked on 2 October; the Overview becomes a list of models with the
   selected model's detail beside it.
2. **The yield-by-laser chart sits above the list, always visible** (not a row in the list).
3. **Title font: Marcellus** ("6", from ten fonts drawn on his own header). Not TMOG's font
   ("im not sure i want to copy the excat fonts"). Body text and numbers stay IBM Plex.
4. **Icons**: downloading allowed ("you can download"); the Tabler set (MIT) — built in 495c046.
5. **Finish-pass screenshots**: yes (done 2026-10-04 on a copy; the list below).

## The screens
**Top bar**: "Laser Trim Analyzer" in Marcellus; Overview · Models · Settings grouped beside it, each
with its icon; the one blue "Process new files" (with its icon) on the right. No big page title
repeated under the bar on any page.

**Overview (B)**, top to bottom:
- the quiet notices and the last run's line, as today;
- one header line: "13 models need a look · $X lost at final test in the last 90 days · newest file
  29 Sep 2026" (dollars = final-test FAILs in the Overview's own 90 calendar days × the model's
  unit price × the cost ratio — the Company trends formula; unpriced models counted, never zero);
- the yield-by-laser chart as a compact strip, always visible;
- the CARD_RULE caption;
- below: a LIST (left, scrolls on its own) and a DETAIL (right):
  - list: "Needs a look (13)" then "Everything else (42)"; each row: model, units, pass %, a
    12-month mini-chart in its laser's colour (lime/teal/purple; final-test-only grey-blue);
    hand-trim tag; "Other models on file (N) ▸" and "Inactive models (N) ▸" folded at the end;
  - detail for the selected row (the first card on load): the model in Marcellus, its status
    word, a pass meter with the %, "was X%", why it is here (red), 12 months of pass %, a facts
    grid (units · lasers · the signal and its baseline → recent · $ lost at final test, 90 days ·
    newest file), and "Open full page ›" (its Summary, charting the card's signal);
- the foot links stay.

**Status bar, every page**: database OK (or what failed) · N models · newest file · drift watch
current (or updating) · files skipped · while a run is going, its progress ("Processing new files
· 34 of 120").

## The finish list (from the 2026-10-04 screenshots)
1. top-bar items spread with gaps, no icons → grouped, with icons (shell);
2. page names repeated as big titles → gone (shell);
3. "Company trends" link vs "Dashboard" page title → "Company trends" everywhere (finish);
4. stale "add prices in Settings → Pricing" → "Settings → Backlog" (finish);
5. three date formats → `gui/v6/formats` (320e2b6) everywhere (all three, in their own files);
6. mixed number precision (0.005201 next to 0.011) → one rule per kind of number (finish);
7. flat text "buttons" and bright blue dropdown arrows → real secondary buttons, theme-coloured
   dropdowns (finish);
8. small centred Model-page tabs → left-aligned, clearer (finish);
9. four lines of red/grey text before the Model-page chart → one headline and a compact facts
   line; the station-spec notice one quiet line that expands (finish);
10. fail-rate chart axis to 125%, rotated "2025-10" labels → capped at 100%, month names (finish);
11. card month bars broken by empty months → mini line charts (Overview B);
12. Process page: run-on line of network paths, two "Process new files" on one screen → one row
    per folder; the page drops its duplicate run button (finish);
13. the dollars hidden on Company trends → on the Overview (Overview B).

## Rules that bind every part
Theme tokens only (no hex outside theme.py); laser names via `laser_label()` (Laser 1 = LTS =
System B; Laser 2 = DLTS = System A; Laser 3 = LTS3 = System C); a failed load is NAMED, never drawn
as nothing or zero; workers never call Tk; dates via `gui/v6/formats`; titles via `theme.title()`;
icons via `gui/v6/icons.icon()`; prices are customer data — real ones never in code, tests,
fixtures or commits (tests invent them); targeted tests only while building (James: the whole suite
and the sweep run once, before the push).
