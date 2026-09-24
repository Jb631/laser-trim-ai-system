# V6 facelift, step 2: the other six pages (2026-09-24)

Step 1 (`2026-09-23-design-system-and-findings-page-design.md`) settled the look — tokens, type,
the building blocks — and rebuilt the Findings page on it. Step 2 relayouts the remaining pages,
which step 1 only restyled. The spec said each page gets "its own short 'what matters on this
page' round … order to be agreed with James after step 1 ships". James's `/goal` of 2026-09-23 —
"finish the redesign … do not pause to ask" — makes each of those rounds a **ruling** here
(*what — why — what it costs if wrong*), for him to overturn after seeing it.

The complaint that started the facelift is the test for every page: *"it's hard to see what's
important."* On each page the most important thing comes first, in words, and everything else is
one step down.

## Rules every page follows (from step 1, plus what step 1 taught)

1. **The page caption is the page's headline, in words** (`PageBase.set_caption`) — e.g. "Holding:
   last lot inside its history · 3 things worth changing".
2. **Sentence case**, verdict badges excepted. **One teal-filled button per screen**, at most.
   Numbers are mono (`theme.mono`). Colours come from `theme.py` only.
3. **A failure is named in a banner (`blocks.banner`, check tone), never drawn as "no data".**
   Every loader on a page is guarded separately; the banner lists which ones failed.
4. **No fixed pixel wraplength on page-width text.** Step 1's audit found all 12 clips at 1280×720
   and none at 1400×900: every one a fixed wraplength wider than the narrow window. Text that
   should wrap follows its container's width through ONE helper in `blocks.py`
   (`wrap_to_width(label, padding)`, bound to the container's `<Configure>`).
5. **The audit is part of done:** `scripts/render_pages.py --audit` at both window sizes, every
   page, every Model tab, clean — plus a look at the pages at work (the audit cannot see colour).
6. Rows use `blocks.row` (model mono · statement + tags · readout mono), headers `blocks.group_header`,
   so lists across the app read the same as the Findings page.

## Order

**Ruling O:** Investigate (the model page) → Home → Settings → Dashboard → Triage → Process — the
sidebar's primary pages first (Home, Investigate, Findings, Settings), the three muted "other
views" last and lightest; within the primary pages, the biggest gain first (Investigate buries its
findings in the seventh tab) — cost if wrong: order only.

## 1. Investigate (the model page)

**What matters:** for one model — is it holding, what is worth changing, and the evidence.
Today: a drift verdict line, two warning captions, metric pills, a σ key, the stats table, the
chart, **seven tabs with Findings last**, the predictor panel.

**Ruling 1:**
1. Caption = the drift-watch verdict sentence (the current `_verdict` label's text).
2. Banners (check tone) for a failed loader and for the trim-vs-final-test spec mismatch — replacing
   the two TIER_WARNING captions.
3. **"Worth changing on this model"** — the model's findings in the same `FindingsView`
   (`include_empty=False`, no Open button), capped to the top three rows with "Show all" and a
   link to the Findings tab; when there are none, one quiet line ("Nothing worth changing stands out
   for this model"). This moves the actionable part from the seventh tab to the top.
4. **"How it's running"** — the metric pills, a one-line σ key (the three-sentence key moves into the
   Drift metrics tab), the stats table.
5. The chart with its Lots/Units toggle; fix the legend covering the "▲ N off-scale" disclosure (seen
   on 8340-1 and 6607) by moving the legend outside the plot area or placing the note above it.
6. Tabs in sentence case, same order: Drift metrics · Smoothness · Units · Final test units · Trim vs
   final test · History · Findings. The Findings tab's facts are restyled with the blocks (step 1's
   spec asked for it; Task 8 kept them as they were).
7. The predictor panel stays at the bottom, restyled, and says in one line what it is: "How well the
   final-test predictor would have called these units (AUC …). It grades nothing."
Also: the raw-datetime bind in `core/yield_stats.py:350` (reached from `_compute_verdict`) is bound
as a string, as step 1 did in the findings engine.
Why: "hard to see what's important" — the findings are the most actionable thing on the page.
Cost if wrong: the page is longer above the chart; the tab still holds everything.

## 2. Home

**What matters:** bring in what is new, then what needs attention. Today: "Bring in what's new"
(one teal button, a specific-folder link, a folders line, progress, a summary line, two notices)
and "What the app is telling you" (the drifting-now list).

**Ruling 2:**
1. Caption: "Last processed {date} · {N} worth changing · {M} drifting now".
2. "Bring in what's new" — unchanged in behaviour, built from the blocks; its notices become quiet
   banners; wraps follow rule 4.
3. **"Worth changing"** — the top three rows of the Findings page's first group across all models
   (`FindingsView`, three rows, "Open Findings" link). Why: Home is the first screen, and the
   findings are what the app now exists to say.
4. **"Drifting now"** — the existing focus list under a specific heading (two "what the app is
   telling you" headings on one page would say nothing).
Cost if wrong: Home is one block longer; the Findings page is unchanged.

## 3. Settings

**What matters:** the one setting Home cannot work without (the ingest folders), then the rest.
Six collapsible cards today, titles in title case.

**Ruling 3:** card titles in sentence case; order: Ingest folders (open) · Backlog — active models
and pricing · Alert thresholds · Per-model specs · ML training · Database; each card's actions use
`primary_button`/`link_button`; destructive actions in Database stay behind their existing
confirmations — cost if wrong: order only.

## 4. Dashboard (an "other view")

**What matters:** is the plant getting better or worse. Today: cost priorities, two yield panels,
the company trend, the lowest-yield list — and **a query failure renders as zeros with no log**.

**Ruling 4:** caption = both yields in words ("Laser 61% · final test 83% over the last 90 days");
**a banner per failed loader** (the silent zeros are a CLAUDE.md violation: "a failure must never
look like a result"); the yield panels' numbers in `SIZE_DISPLAY` mono; the trend; "Where yield is
lost" = the lowest-yield list as `blocks.row`s, beside the cost priorities — cost if wrong: none
beyond layout.

## 5. Triage (an "other view")

**Ruling 5:** "Needs a look" (the focus list) · "All models" (the browse list) as `blocks.row`s; the
scope toggle stays; captions follow rule 4 — cost if wrong: none.

## 6. Process (an "other view")

**Ruling 6:** keep it — one specific folder, one run — restyled: the folder picker, the incremental
checkbox (teal checkmark in `TEXT_INVERSE`), one teal "Start processing", Stop while running,
progress, and after a run a link "See what changed" to Findings — cost if wrong: none.

## Verification

The whole suite; `render_pages.py --audit` clean at both sizes; both QA sweeps on a copy; each
banner and empty state made to fail first; the readability test extended to any new pair; and a
page-by-page list in `BRING_TO_WORK.md` of what to look at on the work laptop.
